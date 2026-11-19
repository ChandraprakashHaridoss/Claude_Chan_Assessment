import streamlit as st
from app_implementation import run_conversation, create_support_ticket, get_ticket_status, vector_index
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="IT Support Bot", page_icon="🤖", layout="wide")

# Simple, clean CSS - minimal styling
st.markdown("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.chat-box { background: white; border-radius: 10px; padding: 20px; margin: 10px 0; }
.user-msg { background: #E8F4F8; padding: 10px 15px; border-radius: 8px; margin: 8px 0; text-align: left; }
.bot-msg { background: #F5F5F5; padding: 10px 15px; border-radius: 8px; margin: 8px 0; text-align: left; }
.badge { display: inline-block; font-size: 0.75em; padding: 3px 8px; background: #E0E0E0; border-radius: 10px; color: #666; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_config" not in st.session_state:
    st.session_state.model_config = {
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 1024
    }

# Header
st.title("🤖 IT Support ChatBot")
st.write("Ask your IT questions and get instant answers")

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎫 Tickets", "⚙️ Settings"])

# ════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════════════════════════
with tab1:
    # Chat history display - only show last message
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        # Show only the most recent message
        msg = st.session_state.chat_history[-1]
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            source = msg.get("source", "Direct")
            st.markdown(f'<div class="bot-msg"><b>Bot:</b> {msg["content"]}<br/><span class="badge">{source}</span></div>', unsafe_allow_html=True)
    else:
        st.info("💡 No messages yet. Start a conversation!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Your question:", placeholder="Ask something...", label_visibility="collapsed")
        with col2:
            submitted = st.form_submit_button("Send", use_container_width=True)
        
        if submitted and user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            try:
                with st.spinner("Thinking..."):
                    response, source = run_conversation(st.session_state.chat_history, st.session_state.model_config)
                st.session_state.chat_history.append({"role": "assistant", "content": response, "source": source})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)[:100]}", "source": "Error"})
            st.rerun()
    
    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Expandable chat history
    with st.expander("📜 View Full History"):
        if st.session_state.chat_history:
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    st.write(f"**{i+1}. You:** {msg['content']}")
                else:
                    source = msg.get("source", "Direct")
                    st.write(f"**{i+1}. Bot:** {msg['content']}")
                    st.caption(f"Source: {source}")
        else:
            st.write("No chat history yet")

# ════════════════════════════════════════════════════════════════
# TAB 2: TICKETS
# ════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create Ticket")
        with st.form("ticket_form"):
            title = st.text_input("Title", placeholder="Issue title")
            category_display = st.selectbox("Category", ["🔐 Security", "📡 Network", "💻 Hardware", "⚙️ Software", "🔧 Other"])
            priority_display = st.radio("Priority", ["Low", "Medium", "High"], horizontal=True)
            desc = st.text_area("Description", placeholder="Describe...", height=80)
            
            if st.form_submit_button("Create", use_container_width=True):
                if title and desc:
                    # Map display names to internal names
                    category_map = {
                        "🔐 Security": "password_reset",
                        "📡 Network": "connectivity",
                        "💻 Hardware": "hardware",
                        "⚙️ Software": "software",
                        "🔧 Other": "other"
                    }
                    priority_map = {
                        "Low": "low",
                        "Medium": "medium",
                        "High": "high"
                    }
                    
                    category = category_map[category_display]
                    priority = priority_map[priority_display]
                    
                    ticket_id = create_support_ticket(category, priority, title, desc)
                    st.success(f"✅ Created: {ticket_id}")
                else:
                    st.error("Fill all fields")
    
    with col2:
        st.subheader("Check Status")
        ticket_id = st.text_input("Ticket ID", placeholder="e.g., TKT-001")
        if st.button("Search", use_container_width=True):
            if ticket_id:
                try:
                    status = get_ticket_status(ticket_id)
                    st.success(f"Status: {status}")
                except Exception as e:
                    st.error(f"Not found: {str(e)[:80]}")

# ════════════════════════════════════════════════════════════════
# TAB 3: SETTINGS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.model_config["model"] = st.selectbox(
            "Model", 
            ["claude-3-5-sonnet-20241022", "claude-3-opus-20250219"],
            index=0
        )
        st.session_state.model_config["temperature"] = st.slider(
            "Temperature", 0.0, 2.0, 0.7, 0.1
        )
    
    with col2:
        st.session_state.model_config["top_p"] = st.slider(
            "Top P", 0.0, 1.0, 1.0, 0.05
        )
        st.session_state.model_config["max_tokens"] = st.slider(
            "Max Tokens", 100, 4096, 1024, 100
        )
    
    st.divider()
    
    st.subheader("📚 Knowledge Base Information")
    
    st.info("📄 **Knowledge Base Source:** `data/it_sector.txt` (1 file)\n\n📊 **Processing:** Document chunked into semantic segments for optimal RAG retrieval")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Chunks", len(vector_index.documents))
        st.metric("Source File", "it_sector.txt")
    
    with col2:
        st.metric("Embedding Model", vector_index.model)
        st.metric("Search Algorithm", "Cosine Similarity")
    
    with st.expander("📖 View All Knowledge Chunks", expanded=False):
        st.write(f"**Total Chunks:** {len(vector_index.documents)}")
        st.write(f"**Embedding Model:** VoyageAI ({vector_index.model})")
        st.write(f"**Search Algorithm:** Cosine Similarity")
        st.write(f"**Chunk Strategy:** Sliding window with 100-char overlap for context preservation")
        st.divider()
        
        for i, doc in enumerate(vector_index.documents, 1):
            with st.container():
                st.write(f"**Chunk {i}** ({len(doc)} chars)")
                st.text_area("Knowledge Chunk", value=doc[:400] + "..." if len(doc) > 400 else doc, height=120, disabled=True, label_visibility="collapsed", key=f"chunk_{i}")
                if i < len(vector_index.documents):
                    st.divider()
    
    st.divider()
    st.write("**About:**")
    st.write("AI-powered IT support assistant with RAG, web search, and ticket management.")
