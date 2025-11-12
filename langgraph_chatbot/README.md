# ZUS Coffee AI Chatbot
An intelligent, full-stack chatbot application for ZUS Coffee built with LangGraph, FastAPI, and RAG architecture. Features product search with vector embeddings, outlet queries with Text2SQL, and an agentic planning system for multi-tool orchestration.

## Features
- **Intelligent Product Search**: FAISS-powered vector search with semantic understanding across 100+ products
- **Natural Language Outlet Queries**: Text2SQL conversion for finding store locations, hours, and contact info
- **Calculator Tool**: Safe arithmetic evaluation for customer convenience
- **Agentic Planning**: LLM-powered decision-making to route queries to the right tool
- **Conversational Memory**: Multi-turn conversations with context persistence
- **Modern Chat UI**: Clean interface with typing indicators and debug mode
- **Robust Error Handling**: Multiple fallback layers and comprehensive input validation
- **Security First**: XSS prevention, SQL injection protection, and input sanitization

## Files Structure
- **part1.py**: It is only include the part 1 of assessment
- **part5.py**: It includes the part 1 to part 5 of assessment, and it is same as backend/backend.py
- **tests/part1_test.py OR tests/part5_test.py**: For test the current part1.py OR part5.py
- **ZUS_PRODUCTS.js**: This is manual retrieval for drinkware products from ZUS website
- **product_ingestion.py**: This is required to run before serving the fastapi_server.py, and it convert the ZUS_PRODUCTS.js, then outputs the "data" folders with FAISS related files
- **fastapi_server.py**: This serves as the API server for the backend, it creates "outlets.db" and load "data" folders; It provides calculator, RAG products search, Text2SQL outlets search tools
- **backend.py**: This serves as the backend server for the frontend, it use agents to plan, chat and to use tools from fastapi_server.py and response to the frontend
- **frontend folder**: This serves as the frontend server to display chatroom UI, when user starts chatting, it sends the request to the backend.py
- **outlets.db**: It contains only 6 data with name (outlet name), location (area), address, opening time, closing time, and phone

## Project Structure
```
langgraph_chatbot/
│── backend.py                # Main Backend          
│── fastapi_server.py         # FastAPI server: Calculator, Products, Outlets APIs
│── outlets.db                # SQLite database
├── frontend/                 # Chat UI
│   ├── index.html            # Main HTML
│   ├── script.js             # Frontend logic
│   └── styles.css            # Styling
├── data/                     # Generated data files
│   ├── products.json         # Processed products
│   ├── products.index        # FAISS vector index
│   └── products_metadata.pkl
├── tests/                    # Test suite
│   ├── part1_test.py         # LangGraph chatbot tests
│   └── part5_test.py         # API and integration tests
├── part1.py                  # LangGraph state machine chatbot
├── part5.py                  # Agentic planner with tool calling
├── product_ingestion.py      # FAISS indexing script
├── ZUS_PRODUCTS.js           # Raw Shopify product data
├── .env                      # Environment variables
└── README.md                 # This file
```

## Setup Instructions

### Prerequisites
- Python
- OpenAI API key

### 1. Download Repository

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-proj-your-api-key-here
BASE_URL=http://localhost:8112
PRODUCTS_RAW_PATH=ZUS_PRODUCTS.js
PRODUCT_INDEX_PATH=data/products.index
PRODUCT_METADATA_PATH=data/products_metadata.pkl
```

### 3. Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

```

### 4. Prepare Product Data (May Skip because already output)

Run the product ingestion script to create FAISS index:

```bash
python product_ingestion.py
```

This will:
- Parse `ZUS_PRODUCTS.js` (ZUS's Shopify product data)
- Generate embeddings using Sentence Transformers
- Create `data/products.index` (FAISS index)
- Save `data/products_metadata.pkl` (product details)

Expected output:
```
Successfully processed products
FAISS index saved to data/products.index
```

### 5. Start API Server

In terminal 1:

```bash
cd server
python fastapi_server.py
```

Server will start on `http://localhost:8112`

Available endpoints:
- `POST /calculator/eval` - Arithmetic evaluation
- `POST /products/search` - Product search with RAG
- `POST /outlets/query` - Outlet queries with Text2SQL

### 6. Start Backend Server

In terminal 2:

```bash
cd backend
python backend.py
```

Server will start on `http://localhost:8111`

Available endpoint:
- `POST /chat` - Main chatbot endpoint

### 7. Open Frontend

Open `frontend/index.html` in your web browser.

Or serve it with Python:

```bash
cd frontend
python -m http.server 8080
```

Then navigate to `http://localhost:8080`

---


## Usage
### Basic Queries
**Product Search:**
```
You: Show me tumblers
You: What coffee machines do you have?
You: Find me merchandise under RM30
```

**Outlet Queries:**
```
You: What time does SS 2 open?
You: Find outlets in Petaling Jaya
You: Which stores are open after 8pm?
```

**Calculator:**
```
You: Calculate 15 * 23 + 100
You: What's 299 / 4?
```

### Command Shortcuts
Type `/` to see autocomplete:
- `/calc <expression>` - Quick calculator
- `/products <query>` - Direct product search
- `/outlets <query>` - Direct outlet query
- `/reset` - Reset chat

### Multi-Command Support
Execute multiple commands in one message:
```
/calc 50 * 3, /products tumblers, /outlets SS 2
```


## Testing
### Run Unit Tests
```bash
# Test LangGraph chatbot (part1)
pytest tests/part1_test.py -v

# Test API server and agentic planner (part5)
pytest tests/part5_test.py -v
```

### Test Coverage
- Slot extraction (LLM + fallback)
- Decision logic for action routing
- API endpoint functionality
- Security: SQL injection, XSS, code injection
- Error handling: API timeouts, malformed inputs
- Multi-turn conversations
---


## Architecture Overview

Frontend (Vercel) -> Backend (Render) -> FastAPI Server (Render)

1. **Frontend**: Send request to backend, and receive request from backend
2. **Backend**: Receive request from frontend, do agentic planning (Extract info, Plan action, Execute action), it calls fastapi server when execute action (not chat function)
3. **FastAPI Server**: Handle request from backend, to identify which tools to use (Calculator, RAG-based product search, Text2SQL Outlet search)

### Data Flow Example: Product Search Query
1. User types: "Show me tumblers under RM50"
   └─> Frontend: POST /chat {"message": "Show me tumblers under RM50"}

2. Backend (port 8111): Receives request, routes to agentic planner
   └─> Creates LangGraph thread, loads conversation history

3. Agentic Planner (part5.py):
   ├─> Extract Slots (LLM): Identifies intent = "product_search"
   ├─> Plan Action (LLM): Decides action = "call_products_api"
   └─> Execute Action: HTTP request to API server

4. API Server (port 8112): POST /products/search
   ├─> Embed query: "tumblers under RM50" → [384-dim vector]
   ├─> FAISS similarity search → Top 5 products with score > 0.5
   ├─> Filter by price: Keep items with price < RM50
   ├─> Format results: Build structured response
   └─> LLM Enhancement: Generate conversational response

5. Response: "I found 3 tumblers under RM50:
   1. All Day Tumbler - RM35.00 (500ml, BPA-free)
   2. Hydro Flask Dupe - RM42.00 (750ml, keeps cold 24hrs)
   3. Stainless Steel Tumbler - RM28.00 (350ml)"

6. Frontend: Displays with typing animation, stores in LocalStorage

### Key Architectural Decisions & Trade-offs
#### 1. LangGraph State Machine vs Traditional Chatbot
**Choice**: LangGraph with explicit state management (location & outlet)

**Pros**:
- Declarative conversation flows with nodes and edges
- Built-in memory persistence with MemorySaver
- Easy to visualize and debug conversation paths
- Supports complex multi-turn conversations

**Cons**:
- More verbose than simple if/else logic for simple bots
- Dependency on LangChain ecosystem

**Trade-off**: Chose complexity for scalability - easier to add new conversation paths as requirements grow.

---

#### 2. FAISS Vector Search vs Traditional Database Search
**Choice**: FAISS with semantic embeddings (all-MiniLM-L6-v2)

**Pros**:
- Semantic search understands intent ("show me tumblers" matches "reusable bottles")
- Sub-millisecond search performance
- Handles typos and synonyms naturally
- No manual keyword mapping required

**Cons**:
- Requires separate embedding model (additional latency during ingestion)
- 384-dimensional vectors use more memory than keyword indices
- Cold start problem - need to pre-compute embeddings
- Less precise for exact SKU/name lookups

**Trade-off**: Sacrificed exact matching precision for natural language understanding. Users prefer "coffee mugs" to "SKU-12345".

---

#### 3. Text2SQL vs Hardcoded Outlet Queries
**Choice**: LLM-powered Text2SQL with validation

**Pros**:
- Handles complex queries ("outlets open after 8pm near KLCC")
- No need to enumerate all possible query patterns
- Natural language interface improves UX
- Easy to extend database schema

**Cons**:
- LLM latency (4-6 seconds per query)
- Potential SQL injection risks (mitigated with validation)
- Inconsistent SQL generation quality
- Higher cost per query vs cached responses

**Trade-off**: Flexibility over speed. Added security layers and fallback validation to minimize SQL injection risk while maintaining natural query capability.

---

#### 4. Microservices (Separate Backend + API Server)
**Choice**: Split into backend chatbot (port 8111) and API server (port 8112)

**Pros**:
- Independent scaling - can scale API server separately
- Separation of concerns - chatbot logic vs data access
- Easier testing - mock API endpoints in chatbot tests
- Multiple frontends can share API server

**Cons**:
- Additional network latency (HTTP calls between services)
- More complex deployment (2 services vs 1)
- Need to handle inter-service communication failures
- Higher infrastructure costs

**Trade-off**: Chose modularity over simplicity. In production, API server can handle multiple chatbot instances and other clients.

---

#### 5. Dual LLM Strategy (Intelligence + Fallback)
**Choice**: LLM for primary logic, keyword matching for fallback

**Pros**:
- Graceful degradation when LLM unavailable
- Reduced API costs for simple queries
- Faster response for common patterns
- Reliability during LLM provider outages

**Cons**:
- Duplicate logic maintenance (LLM prompts + regex patterns)
- Inconsistent quality between LLM and fallback responses
- Harder to test all code paths
- May give "dumb" responses when LLM would have understood

**Trade-off**: Reliability over consistency. Users prefer a basic working bot over error messages during outages.

---

#### 6. Thread-based Memory vs User Sessions
**Choice**: LangGraph thread IDs for conversation persistence

**Pros**:
- Each conversation isolated with unique thread ID
- Easy to implement with LangGraph's MemorySaver
- Can resume conversations across page reloads
- Simpler than user authentication

**Cons**:
- No cross-device conversation sync
- Thread IDs stored in LocalStorage (vulnerable to clearing)
- No user identity - can't attribute conversations to users
- Difficult to implement conversation analytics

**Trade-off**: Chose stateful threads over user management for faster MVP. Can add authentication layer later without rewriting conversation logic.

---
