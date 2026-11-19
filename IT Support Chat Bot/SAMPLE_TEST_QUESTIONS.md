# 🧪 Sample Test Questions for IT Support Chat Bot

**Purpose:** Test all features (RAG, Web Search, MCP) and their combinations

---

## ✅ Question 1: Pure RAG Test
**Tests:** Vector embeddings, FAQ retrieval, Cosine similarity search

```
Question: "How do I reset my password?"
```

**Expected Output:**
- ✅ Direct answer from FAQ (it_sector.txt)
- ✅ No web search needed
- ✅ No ticket created
- 📊 **Component tested:** RAG System

---

## 🔍 Question 2: Web Search Test
**Tests:** Web search fallback, tool calling decision

```
Question: "What are the latest router technologies for enterprise Wi-Fi?"
```

**Expected Output:**
- 🌐 Web search triggered (FAQ doesn't cover latest tech)
- ✅ Latest information from web
- ❌ No ticket should be auto-created
- 📊 **Component tested:** Web Search + Tool Calling

---

## 🎫 Question 3: MCP Ticket Creation Test
**Tests:** MCP server, ticket creation, automatic decision-making

```
Question: "My laptop keeps freezing and crashing. I need urgent help creating a support ticket for this issue."
```

**Expected Output:**
- 📋 Ticket created automatically
- ✅ Ticket ID generated (TICK####)
- 🔴 Priority: high or medium
- 📊 Status: open
- 📊 **Component tested:** MCP Server - create_support_ticket

---

## 🔗 Question 4: Combination Test (RAG + Web Search + MCP)
**Tests:** Multi-step reasoning, combined tools, context awareness

```
Question: "Tell me about Wi-Fi connection issues, the latest troubleshooting methods, and please create a ticket for my ongoing network problems."
```

**Expected Output:**
- 📚 RAG Answer: Retrieved from FAQ (Wi-Fi connection basics)
- 🌐 Web Search: Latest troubleshooting methods
- 🎫 Ticket Created: Support ticket for network issue
- 📊 **Components tested:** RAG + Web Search + MCP (all combined)

---

## 📊 Testing Checklist

| Question | RAG | Web Search | MCP | Expected Result |
|----------|-----|-----------|-----|-----------------|
| Q1 (Reset Password) | ✅ | ❌ | ❌ | FAQ answer only |
| Q2 (Router Tech) | ❌ | ✅ | ❌ | Web search + info |
| Q3 (Ticket Urgent) | ⚠️ | ⚠️ | ✅ | Ticket created |
| Q4 (Combined) | ✅ | ✅ | ✅ | All features |

---

## 🚀 How to Test

1. **Open the app:** `http://localhost:8502`
2. **Copy-paste each question** in the chat box
3. **Observe the output** for each component
4. **Check the System Architecture section** to see which tools were called
5. **Verify ticket creation** shows ticket ID and details

---

## ✨ Success Criteria

- ✅ Q1: Shows FAQ answer directly
- ✅ Q2: Shows web search results
- ✅ Q3: Ticket ID appears (TICK####)
- ✅ Q4: All three components work together

**All tests passing = System is working correctly! 🎉**
