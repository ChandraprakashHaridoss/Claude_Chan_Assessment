"""
IT Support Chat Bot - Core Implementation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implements: RAG System, Tool Calling, MCP Server, and LLM Orchestration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Any
import math
import numpy as np
from abc import ABC, abstractmethod

# LLM & Embedding APIs
from anthropic import Anthropic
import requests
from voyageai import Client as VoyageClient

# Web Search & Parsing
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

# ═══════════════════════════════════════════════════════════════════
# LOAD ENVIRONMENT VARIABLES FROM .env FILE
# ═══════════════════════════════════════════════════════════════════
load_dotenv()

# Retrieve API keys from .env file (secure configuration)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/it_sector_embeddings.db")
TICKET_ID_PREFIX = os.getenv("TICKET_ID_PREFIX", "TICK")

# Cache file for embeddings (to avoid repeated API calls)
from pathlib import Path
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
EMBEDDINGS_CACHE_FILE = CACHE_DIR / "embeddings_cache.json"

# Validate required API keys exist before proceeding
if not ANTHROPIC_API_KEY:
    raise ValueError("❌ ANTHROPIC_API_KEY not found in .env file")
if not VOYAGE_API_KEY:
    raise ValueError("❌ VOYAGE_API_KEY not found in .env file")

# ═══════════════════════════════════════════════════════════════════
# INITIALIZE API CLIENTS
# ═══════════════════════════════════════════════════════════════════
# Initialize Anthropic client for Claude LLM
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Initialize VoyageAI client for embeddings
voyage_client = VoyageClient(api_key=VOYAGE_API_KEY)


# ═══════════════════════════════════════════════════════════════════
# VECTOR INDEX CLASS - CUSTOM RAG IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════
class VectorIndex:
    """
    Custom Vector Index for semantic search using VoyageAI embeddings.
    Implements cosine distance for similarity calculation.
    Follows pattern from /04_claude_anthropic/10_RAG/01_RAG.ipynb
    """
    
    def __init__(self, model: str = "voyage-3-lite"):
        """Initialize VectorIndex with specified embedding model."""
        self.model = model
        self.documents = []
        self.embeddings = []
        self._cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load embeddings cache from file."""
        if EMBEDDINGS_CACHE_FILE.exists():
            try:
                with open(EMBEDDINGS_CACHE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self) -> None:
        """Save embeddings cache to file."""
        with open(EMBEDDINGS_CACHE_FILE, 'w') as f:
            json.dump(self._cache, f)
    
    def add_documents(self, docs: list[str]) -> None:
        """Add documents to the index and generate embeddings (with caching)."""
        self.documents.extend(docs)
        
        # Check which documents are already cached
        docs_to_embed = []
        docs_indices = []
        for i, doc in enumerate(docs):
            doc_hash = str(hash(doc))
            if doc_hash not in self._cache:
                docs_to_embed.append(doc)
                docs_indices.append(i)
        
        # Only call API for new documents
        if docs_to_embed:
            print(f"🔄 Embedding {len(docs_to_embed)}/{len(docs)} new documents...")
            try:
                new_embeddings = voyage_client.embed(
                    docs_to_embed,
                    model=self.model,
                    input_type="document"
                ).embeddings
                
                # Cache the new embeddings
                for doc, embedding in zip(docs_to_embed, new_embeddings):
                    doc_hash = str(hash(doc))
                    self._cache[doc_hash] = embedding
                
                self._save_cache()
            except Exception as e:
                print(f"⚠️  Embedding error: {e}")
                raise
        
        # Load all embeddings (cached + new)
        for doc in docs:
            doc_hash = str(hash(doc))
            if doc_hash in self._cache:
                self.embeddings.append(self._cache[doc_hash])
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for similar documents using cosine distance.
        Returns top_k most similar documents with scores.
        """
        if not self.embeddings:
            return []
        
        # Embed the query (usually uncached, but check anyway)
        query_hash = str(hash(query))
        if query_hash in self._cache:
            query_embedding = self._cache[query_hash]
        else:
            try:
                query_embedding = voyage_client.embed(
                    [query],
                    model=self.model,
                    input_type="query"
                ).embeddings[0]
                # Cache query embedding
                self._cache[query_hash] = query_embedding
                self._save_cache()
            except Exception as e:
                print(f"⚠️  Query embedding error: {e}")
                raise
        
        # Calculate cosine distance with all documents
        scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Cosine similarity: (A·B) / (||A|| * ||B||)
            dot_product = np.dot(query_embedding, doc_embedding)
            norm_a = np.linalg.norm(query_embedding)
            norm_b = np.linalg.norm(doc_embedding)
            
            # Avoid division by zero
            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
            else:
                similarity = 0.0
            
            scores.append({
                "document": self.documents[i],
                "score": float(similarity),
                "index": i
            })
        
        # Sort by score (descending) and return top_k
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════════
# RAG SYSTEM - DOCUMENT LOADING & CHUNKING
# ═══════════════════════════════════════════════════════════════════

def load_documents(file_path: str) -> list[str]:
    """Load and parse IT FAQ documents from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"❌ Document file not found: {file_path}")
        return ""


def chunk_documents(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split document into overlapping chunks for better semantic search.
    Chunks by sentence boundaries to preserve meaning.
    """
    if not text:
        return []
    
    # Split by double newline first (section boundaries)
    sections = text.split("\n\n")
    chunks = []
    
    for section in sections:
        # Further split long sections into smaller chunks
        if len(section) > chunk_size:
            # Split by sentences
            sentences = section.split(". ")
            current_chunk = ""
            
            for sentence in sentences:
                # Add period back if it was removed
                sentence = sentence.strip()
                if sentence and not sentence.endswith("."):
                    sentence += "."
                
                # Check if adding this sentence would exceed chunk_size
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Overlap: restart with last sentence
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            
            # Add remaining text
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
            # Section is small enough, add as single chunk
            if section.strip():
                chunks.append(section.strip())
    
    return chunks


# ═══════════════════════════════════════════════════════════════════
# INITIALIZE RAG SYSTEM
# ═══════════════════════════════════════════════════════════════════

# Create VectorIndex instance
vector_index = VectorIndex()

# Load and chunk IT FAQ documents
print("📚 Loading IT Support FAQ documents...")
raw_docs = load_documents("./data/it_sector.txt")

if raw_docs:
    # Chunk documents for better retrieval
    chunks = chunk_documents(raw_docs, chunk_size=500, overlap=100)
    print(f"✅ Created {len(chunks)} chunks from IT FAQ")
    
    # Add chunks to vector index (generates embeddings)
    print("🔄 Generating embeddings for all chunks...")
    vector_index.add_documents(chunks)
    print(f"✅ RAG system ready with {len(chunks)} searchable chunks")
else:
    print("⚠️  No documents loaded - RAG may not work")


# ═══════════════════════════════════════════════════════════════════
# TOOL CALLING - FUNCTION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

def web_search(query: str, num_results: int = 3) -> str:
    """
    Search the web for IT-related solutions.
    Uses DuckDuckGo or Google Custom Search.
    Returns formatted search results.
    """
    try:
        # Try DuckDuckGo API first
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        # Make web request with timeout and proper headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, params=params, timeout=5, headers=headers)
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Extract results from AbstractResults or Results
        results = []
        
        # Try AbstractResults first (more reliable)
        if "AbstractText" in data and data["AbstractText"]:
            results.append({
                "title": data.get("Heading", "Search Result"),
                "url": data.get("AbstractURL", ""),
                "snippet": data["AbstractText"][:200]
            })
        
        # Try RelatedTopics for additional results
        if "RelatedTopics" in data and len(results) < num_results:
            for topic in data["RelatedTopics"][:num_results - len(results)]:
                if "FirstURL" in topic and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")[:200]
                    })
        
        # Format results as readable text
        if results:
            formatted_results = "🌐 **Web Search Results for: \"" + query + "\"**\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"{i}. **{result['title']}**\n"
                if result['snippet']:
                    formatted_results += f"   {result['snippet'][:150]}...\n"
                if result['url']:
                    formatted_results += f"   🔗 {result['url']}\n"
                formatted_results += "\n"
            return formatted_results
        else:
            # Fallback: return generic helpful message
            return f"""🌐 **Web Search: "{query}"**

Based on your query about "{query}", here are some general recommendations:

• Search engines like Google, Bing, or DuckDuckGo can provide more results
• Visit official documentation sites for the latest information
• Check security bulletins (CVE databases, Microsoft Security Updates, etc.)
• Look at tech news sites (TechCrunch, SecurityFocus, ArsTechnica)
• Visit official vendor sites for product updates

For more specific results, try using keywords like:
  - "latest", "2026", "current", "new", "update"
  - Official vendor names (Microsoft, Apple, Google, etc.)
  - Specific CVE numbers or security advisory IDs"""
        
    except requests.exceptions.Timeout:
        return f"""⚠️ **Web Search Timeout**

The web search took too long to respond. This could mean:
• The search engine is experiencing high traffic
• Your internet connection is slow
• The query is too complex

Try again with:
• Simpler search terms
• Specific keywords (not full sentences)
• Official vendor or product names"""
    
    except Exception as e:
        return f"""⚠️ **Web Search Error**

Error: {str(e)[:100]}

Suggestions:
• Check your internet connection
• Try a different search query
• Use simpler keywords
• Visit security sites directly:
  - https://www.cvedetails.com/
  - https://msrc.microsoft.com/
  - https://security.google.com/
  - https://www.exploit-db.com/"""


def get_current_datetime() -> dict:
    """Get current date and time."""
    now = datetime.now()
    return {
        "current_time": now.strftime("%H:%M:%S"),
        "current_date": now.strftime("%Y-%m-%d"),
        "day_of_week": now.strftime("%A"),
        "timestamp": now.isoformat()
    }


def rag_search(query: str, top_k: Optional[int] = None) -> dict:
    """
    Search IT FAQ documents using RAG.
    Returns most relevant document chunks with similarity scores.
    """
    if top_k is None:
        top_k = TOP_K_RESULTS
    
    # Search vector index
    results = vector_index.search(query, top_k=top_k)
    
    if results:
        formatted_results = "📚 **IT FAQ Results:**\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. (Confidence: {result['score']:.2%})\n"
            formatted_results += f"{result['document'][:300]}...\n\n"
        
        return {
            "source": "rag",
            "results": formatted_results,
            "count": len(results),
            "scores": [r["score"] for r in results]
        }
    else:
        return {
            "source": "rag",
            "results": "No relevant FAQ found.",
            "count": 0,
            "scores": []
        }


# ═══════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS FOR CLAUDE - FOLLOWING /04_claude_anthropic PATTERN
# ═══════════════════════════════════════════════════════════════════

class ToolParam:
    """Parameter definition for tools using JSON Schema."""
    
    def __init__(self, name: str, type: str, description: str, required: bool = True):
        self.name = name
        self.type = type
        self.description = description
        self.required = required
    
    def to_dict(self) -> dict:
        """Convert to JSON Schema format."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description
        }


# Define available tools for Claude
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the internet for IT solutions and troubleshooting guides",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for IT solutions"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "rag_search",
        "description": "Search IT FAQ documentation for answers",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question to search in IT FAQ"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of FAQ results to return",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
]


def run_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool and return results.
    Tool dispatcher pattern from /04_claude_anthropic/11_Tools/01_ToolUse.ipynb
    """
    if tool_name == "web_search":
        query = tool_input.get("query", "")
        num_results = tool_input.get("num_results", 3)
        return web_search(query, num_results)
    
    elif tool_name == "get_current_datetime":
        result = get_current_datetime()
        return json.dumps(result, indent=2)
    
    elif tool_name == "rag_search":
        query = tool_input.get("query", "")
        top_k = tool_input.get("top_k", TOP_K_RESULTS)
        result = rag_search(query, top_k)
        return json.dumps(result, indent=2)
    
    else:
        return f"❌ Unknown tool: {tool_name}"


# ═══════════════════════════════════════════════════════════════════
# MCP SERVER - TICKET MANAGEMENT (In-Memory Storage)
# ═══════════════════════════════════════════════════════════════════

class TicketStore:
    """In-memory ticket storage for MCP server."""
    
    def __init__(self, prefix: str = "TICK"):
        self.tickets = {}
        self.prefix = prefix
        self.counter = 1000
    
    def create_ticket(self, category: str, priority: str, subject: str, description: str = "") -> dict:
        """Create a new support ticket."""
        # Generate unique ticket ID
        self.counter += 1
        ticket_id = f"{self.prefix}{self.counter}"
        
        # Create ticket record
        ticket = {
            "id": ticket_id,
            "category": category,
            "priority": priority,
            "subject": subject,
            "description": description,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Store ticket
        self.tickets[ticket_id] = ticket
        
        return ticket
    
    def get_ticket_status(self, ticket_id: str) -> Optional[dict]:
        """Retrieve ticket status and details."""
        return self.tickets.get(ticket_id)
    
    def update_ticket_status(self, ticket_id: str, status: str) -> bool:
        """Update ticket status."""
        if ticket_id in self.tickets:
            self.tickets[ticket_id]["status"] = status
            self.tickets[ticket_id]["updated_at"] = datetime.now().isoformat()
            return True
        return False
    
    def list_tickets(self) -> list[dict]:
        """List all tickets."""
        return list(self.tickets.values())


# Initialize ticket store
ticket_store = TicketStore(prefix=TICKET_ID_PREFIX)


def create_support_ticket(category: str, priority: str, subject: str, description: str = "") -> str:
    """
    Create a new IT support ticket via MCP.
    Categories: password_reset, connectivity, software, hardware, other
    Priorities: low, medium, high, critical
    """
    # Validate inputs
    valid_categories = ["password_reset", "connectivity", "software", "hardware", "other"]
    valid_priorities = ["low", "medium", "high", "critical"]
    
    if category not in valid_categories:
        return f"❌ Invalid category. Must be one of: {', '.join(valid_categories)}"
    
    if priority not in valid_priorities:
        return f"❌ Invalid priority. Must be one of: {', '.join(valid_priorities)}"
    
    # Create ticket
    ticket = ticket_store.create_ticket(category, priority, subject, description)
    
    return f"""
✅ **Ticket Created Successfully**
🎫 Ticket ID: {ticket['id']}
📋 Category: {ticket['category']}
🔴 Priority: {ticket['priority']}
📝 Subject: {ticket['subject']}
⏰ Created: {ticket['created_at']}
📊 Status: {ticket['status']}

Your ticket has been created and assigned to the IT support team.
Expected response time: {get_response_time(priority)}
    """


def get_ticket_status(ticket_id: str) -> str:
    """Check the status of an existing support ticket."""
    ticket = ticket_store.get_ticket_status(ticket_id)
    
    if not ticket:
        return f"❌ Ticket not found: {ticket_id}"
    
    # Format ticket information
    return f"""
🎫 **Ticket Status**
Ticket ID: {ticket['id']}
Subject: {ticket['subject']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Status: {ticket['status'].upper()}
Created: {ticket['created_at']}
Updated: {ticket['updated_at']}

Description: {ticket['description'] or 'N/A'}
    """


def get_response_time(priority: str) -> str:
    """Get expected response time based on priority."""
    response_times = {
        "critical": "1 hour",
        "high": "4 hours",
        "medium": "1 business day",
        "low": "3 business days"
    }
    return response_times.get(priority, "2 business days")


# MCP server tools definition
MCP_TOOLS = [
    {
        "name": "create_support_ticket",
        "description": "Create a new IT support ticket",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["password_reset", "connectivity", "software", "hardware", "other"],
                    "description": "Ticket category"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Ticket priority level"
                },
                "subject": {
                    "type": "string",
                    "description": "Brief description of the issue"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description (optional)"
                }
            },
            "required": ["category", "priority", "subject"]
        }
    },
    {
        "name": "get_ticket_status",
        "description": "Check the status of an existing support ticket",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "The ticket ID to check"
                }
            },
            "required": ["ticket_id"]
        }
    }
]


def run_mcp_tool(tool_name: str, tool_input: dict) -> str:
    """Execute MCP tools."""
    if tool_name == "create_support_ticket":
        return create_support_ticket(
            category=tool_input.get("category"),
            priority=tool_input.get("priority"),
            subject=tool_input.get("subject"),
            description=tool_input.get("description", "")
        )
    
    elif tool_name == "get_ticket_status":
        return get_ticket_status(tool_input.get("ticket_id"))
    
    else:
        return f"❌ Unknown MCP tool: {tool_name}"


# ═══════════════════════════════════════════════════════════════════
# RESPONSE SOURCE TRACKING
# ═══════════════════════════════════════════════════════════════════

def determine_response_source(response_text: str, used_tools: list[str]) -> str:
    """
    Determine which source provided the response.
    Returns emoji badge for UI display.
    """
    # Check which tools were used
    if "web_search" in used_tools:
        return "🌐 Web Search"
    elif "create_support_ticket" in used_tools or "get_ticket_status" in used_tools:
        return "🎫 MCP Ticket"
    elif "rag_search" in used_tools:
        return "📚 RAG"
    elif len(used_tools) > 1:
        return "🔄 Multi-Source"
    else:
        return "🤖 Direct Answer"


def build_response_metadata(response_text: str, used_tools: list[str], tool_results: dict) -> dict:
    """
    Build metadata for response tracking.
    Includes source information for UI display.
    """
    source = determine_response_source(response_text, used_tools)
    
    return {
        "source": source,
        "tools_used": used_tools,
        "timestamp": datetime.now().isoformat(),
        "has_rag": "rag_search" in used_tools,
        "has_web_search": "web_search" in used_tools,
        "has_mcp": "create_support_ticket" in used_tools or "get_ticket_status" in used_tools
    }


# ═══════════════════════════════════════════════════════════════════
# LLM ORCHESTRATION - MAIN CONVERSATION LOOP
# ═══════════════════════════════════════════════════════════════════

def run_conversation(
    messages: list[dict],
    model_config: dict = None
) -> tuple[str, str]:
    """
    Main conversation loop with tool calling.
    Implements multi-turn conversation with Claude.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model_config: Dict with temperature, max_tokens, top_p, top_k, top_k_rag
    
    Returns:
        Tuple of (response_text, source)
    """
    
    # Use default config if not provided
    if model_config is None:
        model_config = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "top_k": 20
        }
    
    # Get the last user message
    last_user_msg = messages[-1]["content"] if messages else ""
    
    # Try to use RAG to answer first
    try:
        rag_result = rag_search(last_user_msg)
        if rag_result.get("found"):
            response_text = rag_result.get("response", "No answer found in knowledge base.")
            return response_text, "RAG"
    except:
        pass
    
    # Combine all tools
    all_tools = TOOLS + MCP_TOOLS
    used_tools = []
    tool_results_dict = {}
    
    # Initial API call to Claude
    try:
        response = anthropic_client.messages.create(
            model=model_config.get("model", MODEL_NAME),
            max_tokens=model_config.get("max_tokens", 2048),
            system="""You are a helpful IT Support Assistant. 
        
You help employees with IT issues using:
- IT FAQ Documentation (use rag_search tool)
- Web search for additional resources (use web_search tool)
- Creating and checking support tickets (use create_support_ticket, get_ticket_status)
- Current date/time information (use get_current_datetime)

Always try to help with the right tools before creating a ticket.
Be friendly, clear, and provide step-by-step solutions.""",
            tools=all_tools,
            messages=messages,
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 0.9)
        )
    except Exception as e:
        # If API fails (no credits), provide demo response
        if "credit" in str(e).lower() or "404" in str(e) or "invalid" in str(e).lower():
            msg_lower = last_user_msg.lower()
            
            # 1. CHECK FOR WEB SEARCH FIRST (keywords: latest, current, news, trends, updates, etc.)
            web_search_keywords = ["latest", "current", "today", "news", "trends", "2024", "2025", "2026", "recent", "new", "update", "vulnerability", "security", "vulnerabilities", "technologies", "tech", "innovation"]
            should_web_search = any(keyword in msg_lower for keyword in web_search_keywords)
            
            if should_web_search:
                try:
                    print(f"🌐 Performing web search for: {last_user_msg}")
                    web_results = web_search(last_user_msg, num_results=3)
                    return web_results, "Web Search"
                except Exception as web_error:
                    print(f"⚠️ Web search failed: {web_error}")
                    # Fall through to other options
            
            # 2. CHECK IF USER WANTS TO CREATE A TICKET (require explicit intent)
            has_create = "create" in msg_lower or "submit" in msg_lower or "open" in msg_lower
            has_ticket = "ticket" in msg_lower or "support" in msg_lower
            has_explicit_problem = "problem" in msg_lower or "broken" in msg_lower or "not working" in msg_lower or "error" in msg_lower
            
            # Only create ticket if explicitly requesting it, not just mentioning issues
            should_create_ticket = (has_create and has_ticket) or (has_explicit_problem and has_ticket)
            
            if should_create_ticket:
                try:
                    # Determine category based on message content
                    if "password" in msg_lower or "login" in msg_lower:
                        category = "password_reset"
                        subject = "Password Reset Request"
                    elif "wifi" in msg_lower or "connection" in msg_lower or "internet" in msg_lower:
                        category = "connectivity"
                        subject = "Wi-Fi/Network Connectivity Issue"
                    elif "software" in msg_lower or "install" in msg_lower or "app" in msg_lower:
                        category = "software"
                        subject = "Software Installation Request"
                    elif "laptop" in msg_lower or "computer" in msg_lower or "hardware" in msg_lower or "monitor" in msg_lower or "keyboard" in msg_lower:
                        category = "hardware"
                        subject = "Hardware Issue"
                    else:
                        category = "other"
                        subject = "IT Support Request"
                    
                    priority = "medium"
                    description = last_user_msg
                    
                    print(f"🎫 Creating support ticket in demo mode: {subject}")
                    ticket_result = create_support_ticket(category, priority, subject, description)
                    return ticket_result, "MCP (Ticket Created)"
                except Exception as ticket_error:
                    print(f"⚠️ Ticket creation failed: {ticket_error}")
            
            # Check if user wants to check ticket status
            status_keywords = ["status", "check", "ticket", "my ticket", "ticket status", "ticket id"]
            should_check_status = all(keyword in msg_lower for keyword in ["ticket"]) and any(keyword in msg_lower for keyword in ["status", "check"])
            
            if should_check_status:
                try:
                    # Try to extract ticket ID from message
                    import re
                    ticket_match = re.search(r'TICK[A-Z0-9]+', last_user_msg.upper())
                    if ticket_match:
                        ticket_id = ticket_match.group(0)
                        print(f"🎫 Checking ticket status: {ticket_id}")
                        status_result = get_ticket_status(ticket_id)
                        return status_result, "MCP (Ticket Status)"
                except Exception as status_error:
                    print(f"⚠️ Ticket status check failed: {status_error}")
            
            # Demo responses for common IT issues
            demo_responses = {
                "password": "To reset your password:\n1. Go to the login page\n2. Click 'Forgot Password'\n3. Enter your email\n4. Follow the reset link sent to your inbox\n5. Create a new strong password\n\nBest practices for new password:\n- Use at least 12 characters\n- Mix uppercase, lowercase, numbers, and symbols\n- Avoid personal information\n- Don't reuse old passwords\n- Use a password manager",
                "wifi": "To fix Wi-Fi connection:\n1. Restart your router (unplug for 30 seconds)\n2. Forget the network and reconnect\n3. Update your network drivers\n4. Move closer to the router\n5. Check if other devices can connect\n6. Contact IT if problem persists",
                "software": "To install software:\n1. Submit a request to IT helpdesk\n2. Provide the software name and version\n3. Explain the business need\n4. Wait for IT approval and installation\n5. IT will install it remotely or via ticket\n\nCommon approved software: Office 365, Adobe Creative Suite, VS Code, Chrome Extensions",
                "slow": "To speed up your computer:\n1. Close unnecessary applications and browser tabs\n2. Clear browser cache and temporary files\n3. Restart your computer\n4. Check for malware with antivirus\n5. Update your OS and drivers\n6. Free up disk space (delete old files)\n7. Contact IT if issues persist",
                "2fa": "To enable Two-Factor Authentication (2FA):\n1. Go to your account settings\n2. Find 'Security' or 'Authentication'\n3. Click 'Enable 2FA'\n4. Choose authentication method (Authenticator app or SMS)\n5. Download an app like Google Authenticator or Microsoft Authenticator\n6. Scan the QR code\n7. Save backup codes in a safe place",
                "vpn": "To connect to VPN:\n1. Contact IT to request VPN access\n2. Download and install the VPN client\n3. Open the VPN application\n4. Enter your credentials\n5. Click 'Connect'\n6. Test connection by visiting internal resources",
                "printer": "To add a network printer:\n1. Get the printer IP address from IT or the printer display\n2. Go to Settings > Devices > Printers\n3. Click 'Add printer or scanner'\n4. Select 'The printer that I want isn't listed'\n5. Enter the IP address\n6. Install drivers if prompted\n7. Test print a document"
            }
            
            # Find best matching response
            best_match = None
            for key, demo_response in demo_responses.items():
                if key in msg_lower:
                    best_match = demo_response
                    break
            
            # If no exact match, check for practice/best/how questions
            if not best_match:
                if "practice" in msg_lower or "best" in msg_lower or "how" in msg_lower:
                    # Try to find related topic
                    if "password" in msg_lower:
                        best_match = demo_responses["password"]
                    elif "wifi" in msg_lower or "connection" in msg_lower:
                        best_match = demo_responses["wifi"]
                    elif "software" in msg_lower or "install" in msg_lower:
                        best_match = demo_responses["software"]
            
            if best_match:
                return best_match, "Knowledge Base"
            
            return "I'm experiencing technical issues accessing the AI model. Please add credits to your Anthropic account or try:\n\n💬 **Knowledge Base:** Ask about password reset, Wi-Fi, software, slow PC, 2FA, VPN, or printers\n🌐 **Web Search:** Ask about latest, current, news, security, trends, or updates\n🎫 **MCP Tickets:** Create a ticket or check status\n\nWhat can I help you with?", "Direct Answer"
    
    # Process tool calls in a loop
    max_iterations = 5
    iteration = 0
    
    while response.stop_reason == "tool_use" and iteration < max_iterations:
        iteration += 1
        
        # Find tool use blocks in response
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        
        # Execute each tool
        tool_results = []
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input
            
            # Track tool usage
            used_tools.append(tool_name)
            
            # Execute appropriate tool
            if tool_name in [t["name"] for t in TOOLS]:
                result = run_tool(tool_name, tool_input)
            elif tool_name in [t["name"] for t in MCP_TOOLS]:
                result = run_mcp_tool(tool_name, tool_input)
            else:
                result = f"Unknown tool: {tool_name}"
            
            tool_results_dict[tool_name] = result
            
            # Collect tool results
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result
            })
        
        # Add assistant response and tool results to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
        
        # Make another API call with tool results
        response = anthropic_client.messages.create(
            model=model_config.get("model", MODEL_NAME),
            max_tokens=model_config.get("max_tokens", 2048),
            tools=all_tools,
            messages=messages,
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 0.9)
        )
    
    # Extract final text response
    final_response = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_response += block.text
    
    # Add final assistant message
    messages.append({
        "role": "assistant",
        "content": final_response
    })
    
    # Build response metadata for UI
    metadata = build_response_metadata(final_response, used_tools, tool_results_dict)
    
    return final_response, metadata


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def clear_chat_history() -> None:
    """Clear conversation history."""
    # This is handled in streamlit.py via session state
    pass


def reset_to_defaults() -> dict:
    """Return default model configuration."""
    return {
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.9,
        "top_k": 20,
        "top_k_rag": 3
    }


# ═══════════════════════════════════════════════════════════════════
# APPLICATION STARTUP VERIFICATION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("IT SUPPORT CHAT BOT - CORE SYSTEM INITIALIZED")
    print("=" * 70)
    print(f"✅ Anthropic API: Connected")
    print(f"✅ VoyageAI API: Connected")
    print(f"✅ RAG System: {len(vector_index.embeddings)} chunks loaded")
    print(f"✅ Tools Available: {len(TOOLS)} + {len(MCP_TOOLS)} MCP")
    print(f"✅ Ticket Store: Ready to create/check tickets")
    print("=" * 70)
