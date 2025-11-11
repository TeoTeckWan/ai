from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from datetime import datetime
from requests.exceptions import RequestException, Timeout, ConnectionError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import operator
import requests
import uvicorn


load_dotenv()
api_key_openai = os.getenv("OPENAI_API_KEY")
api_server_url = os.getenv("BASE_URL")

llm = ChatOpenAI(
    model="gpt-5-nano-2025-08-07", 
    temperature=0,  # Deterministic for intent parsing
    api_key=api_key_openai
)

API_TIMEOUT = 30

# STEP 1: State Schema for Agentic Planning
class AgentState(TypedDict):
    """
    """

    messages:    Annotated[List[BaseMessage], operator.add]
    location:    Optional[str]
    outlet_name: Optional[str]
    query_type:  Optional[str]
    
    action_type:          Optional[str]  # 'ask', 'call_tool', 'finish'
    action_history:       Annotated[List[Dict], operator.add]  # Log of decisions
    tool_calls:           List[str]  # List of tools used
    pending_action:       Optional[Dict[str, Any]]  # Action to execute
    last_decision_reason: Optional[str]  # Explanation of decision

    api_errors:  List[Dict[str, Any]]  # Track API failures



# STEP 2: Extract Slots 
def extract_slots(state: AgentState) -> AgentState:
    """
    Extract information from user message with outlet data validation.
    """

    last_message = state["messages"][-1].content
    current_location = state.get("location")
    current_outlet = state.get("outlet_name")

    print(f"\n🔍 EXTRACT_SLOTS - Input State:")
    print(f"   Current Location: {current_location}")
    print(f"   Current Outlet: {current_outlet}")
    print(f"   User Message: {last_message}")
    
    prompt = f"""Extract structured information from the user's message.

**Current Context:**
- Location: {current_location or "unknown"}
- Outlet: {current_outlet or "unknown"}

**User Message:** "{last_message}"

**Extract:**
1. **location**: City/area (Petaling Jaya, Kuala Lumpur, etc.)
2. **outlet_name**: Specific outlet name
3. **query_type**: User intent
   - "calculation" - math expressions
   - "opening_time" / "closing_time" / "address" / "check_outlet" - outlet queries
   - "product_query" - products/drinkware
   - "general_chat" - greetings, thank you, general questions
   - null - unclear

**Rules:**
- If user mentions a NEW outlet name, extract it and set location to null (let API determine location)
- If user asks follow-up questions without mentioning outlet/location, preserve existing context
- Product queries don't need location/outlet
- General chat (hi, hello, thank you, how are you) should have query_type="general_chat"
- When a new outlet OR location is mentioned, REPLACE the old values

Return ONLY valid JSON:
{{"location": "..." or null, "outlet_name": "..." or null, "query_type": "..." or null}}
"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert at extracting structured information from conversational text."),
            HumanMessage(content=prompt)
        ])

        # Parse LLM response
        content = response.content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            # Remove markdown code blocks
            content = content.replace("```json", "").replace("```", "").strip()

        extracted = json.loads(content)

        new_outlet = extracted.get("outlet_name")
        new_location = extracted.get("location")
        new_query_type = extracted.get("query_type") or state.get("query_type")

        print(f"   ✅ Extracted from LLM: outlet={extracted.get('outlet_name')}, location={extracted.get('location')}")
        print(f"   📝 Final State: outlet={new_outlet}, location={new_location}")

        return {
            "location": new_location,
            "outlet_name": new_outlet,
            "query_type": new_query_type
        }
        
    except Exception as e:
        # Fallback: preserve existing state
        return {
            "location": current_location,
            "outlet_name": current_outlet,
            "query_type": state.get("query_type")
        }



# STEP 3: PLANNER - The Brain of the Agent
def plan_next_action(state: AgentState) -> Dict[str, Any]:
    """
    Decides what the agent should do next.
    This is the "controller loop" that analyzes the current state and determines the next action.
    
    Decision Tree:
    1. Check if general chat -> chat
    2. Check if calculation -> call_calculator
    3. Check if product query -> call_products
    4. Check if outlet query -> call_outlets
    5. Otherwise -> chat (default to friendly response)
    
    Args:
        state: Current agent state with all context
        
    Returns:
        dict with action_type and reasoning
    """
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # Create planning prompt for LLM
    planning_prompt = f"""You are a planning agent for a coffee shop chatbot.

User's latest message: "{last_message}"

Current context:
- Location: {state.get('location')}
- Outlet: {state.get('outlet_name')}
- Query type / Intent: {state.get('query_type')}

Available actions:
1. "call_calculator" - Math/calculation queries
2. "call_products" - Product queries (uses RAG API)
3. "call_outlets" - Outlet queries (uses Text2SQL API)
4. "chat" - General conversation (greetings, thank you, general questions, chitchat)
5. "finish" - Query complete (use sparingly, prefer "chat" for responses)

Important:
- Product queries should use "call_products"
- Outlet queries should use "call_outlets"
- Greetings, thank yous, how are you, general questions -> use "chat"
- Default to "chat" when in doubt to be friendly
- Use "finish" ONLY when user explicitly says bye/goodbye

Analyze the user's intent and decide which action to take.

Return ONLY valid JSON:
{{"action": "...", "reason": "brief explanation"}}
"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert planning agent."),
            HumanMessage(content=planning_prompt)
        ])
        
        decision = json.loads(response.content)
        action = decision.get("action", "finish")
        reason = decision.get("reason", "No specific reason")
        
        return {
            "action_type": action,
            "last_decision_reason": reason,
            "pending_action": {
                "type": action,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"   >  Planner LLM error: {e}")
        # Fallback to rule-based planning
        return plan_next_action_fallback(state)

def plan_next_action_fallback(state: AgentState) -> Dict[str, Any]:
    """
    Fallback planner using rule-based logic.
    Used if LLM planning fails.
    """
    last_message = state["messages"][-1].content.lower() if state["messages"] else ""
    query_type = state.get("query_type")
    
    # Check for general chat
    if query_type == "general_chat" or any(word in last_message for word in ["hi", "hello", "hey", "thank", "how are you", "what can you do", "help"]):
        return {
            "action_type": "chat",
            "last_decision_reason": "General conversation detected",
            "pending_action": {"type": "chat"}
        }
    
    # Check for calculator intent
    if any(word in last_message for word in ["calculate", "plus", "minus", "+", "-", "*", "/"]):
        return {
            "action_type": "call_calculator",
            "last_decision_reason": "Arithmetic intent detected",
            "pending_action": {"type": "call_calculator"}
        }
    
    # Check for product intent
    if query_type == "product_query" or any(word in last_message for word in ["product", "tumbler", "mug", "drinkware", "cup", "merchandise"]):
        return {
            "action_type": "call_products",
            "last_decision_reason": "Product query detected",
            "pending_action": {"type": "call_products"}
        }
    
    # Check for outlet intent
    if any(word in last_message for word in ["outlet", "open", "close", "where", "address", "location", "find", "store"]):
        return {
            "action_type": "call_outlets",
            "last_decision_reason": "Outlet query detected",
            "pending_action": {"type": "call_outlets"}
        }
    
    # Default: chat 
    return {
        "action_type": "chat",
        "last_decision_reason": "Default to friendly chat",
        "pending_action": {"type": "chat"}
    }


# STEP 4: TOOLS / API
def call_calculator_api(expression: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Call the Calculator API endpoint.
    
    Args:
        expression: Mathematical expression (e.g., "5+5", "10*2")
        timeout: Request timeout in seconds
        
    Returns:
        dict with result or error details
        
    Error Handling:
    - Connection errors (API down)
    - Timeout errors (API too slow)
    - HTTP errors (500, 404, etc.)
    - Invalid response format
    - Malformed expressions
    """
    
    # Use provided timeout or default to global API_TIMEOUT
    request_timeout = timeout if timeout is not None else API_TIMEOUT

    # print(f"\n🔧 TOOL: Calculator API")
    # print(f"   Expression: {expression}")
    # print(f"   Endpoint: {api_server_url}/calculate")

    try:
        response = requests.post(
            f"{api_server_url}/calculate",
            json={"expression": expression},
            timeout=request_timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "result": data.get("result"),
                "expression": expression,
                "error_type": None,
                "status_code": 200,
                "used_fallback": False
            }
        
        elif response.status_code == 422:
            error_data = response.json()
            detail = error_data.get("detail", "Invalid expression")
            
            if isinstance(detail, list) and len(detail) > 0:
                error_msg = detail[0].get("msg", "Invalid expression format")
            else:
                error_msg = str(detail)
            
            return {
                "success": False,
                "result": None,
                "expression": expression,
                "error_type": "validation_error",
                "error_message": error_msg,
                "status_code": 422,
                "used_fallback": False
            }
        
        else:
            return {
                "success": False,
                "result": None,
                "expression": expression,
                "error_type": "server_error",
                "error_message": f"Status {response.status_code}",
                "status_code": response.status_code,
                "used_fallback": False
            }
    
    except (Timeout, ConnectionError):
        return calculate_locally(expression)
    
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "expression": expression,
            "error_type": "unknown_error",
            "error_message": str(e),
            "status_code": None,
            "used_fallback": False
        }

# Fallback condition for calculator API
def calculate_locally(expression: str) -> Dict[str, Any]:
    """
    Fallback calculator that works locally without API.
    
    Used when:
    - API is completely down
    - All retries failed
    - User explicitly requests fallback
    
    Args:
        expression: Math expression
        
    Returns:
        Calculation result dict
    """
    
    try:
        # Safety: only allow basic math operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return {
                "success": False,
                "result": None,
                "expression": expression,
                "error_type": "validation_error",
                "error_message": "Invalid characters in expression",
                "used_fallback": True
            }
        
        # Evaluate safely
        result = eval(expression, {"__builtins__": {}}, {})
        
        return {
            "success": True,
            "result": result,
            "expression": expression,
            "error_type": None,
            "error_message": None,
            "used_fallback": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "expression": expression,
            "error_type": "evaluation_error",
            "error_message": str(e),
            "used_fallback": True
        }



def call_products_api(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Call Products RAG API
    
    GET /products?query={query}&top_k={top_k}
    """
    
    try:
        response = requests.get(
            f"{api_server_url}/products",
            params={"query": query, "top_k": top_k},
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            
            return {
                "success": True,
                "products": data["results"],
                "count": data["count"],
                "summary": data["summary"],
                "ai_response": data["ai_response"],
                "error_type": None
            }
        
        elif response.status_code == 400:
            error_data = response.json()
            print(f"   - Validation error")
            
            return {
                "success": False,
                "error_type": "validation_error",
                "error_message": error_data.get("detail", "Invalid query")
            }
        
        else:
            print(f"   - Status: {response.status_code}")
            
            return {
                "success": False,
                "error_type": "server_error",
                "error_message": f"API returned {response.status_code}"
            }
    
    except Timeout:
        print(f"   - Request timeout")
        
        return {
            "success": False,
            "error_type": "timeout",
            "error_message": "Products API timeout"
        }
    
    except ConnectionError:
        print(f"   - Connection failed")
        
        return {
            "success": False,
            "error_type": "connection_error",
            "error_message": "Cannot connect to Products API"
        }
    
    except Exception as e:
        print(f"   - Error: {e}")
        
        return {
            "success": False,
            "error_type": "unknown_error",
            "error_message": str(e)
        }



def call_outlets_api(query: str) -> Dict[str, Any]:
    """
    Call Outlets Text2SQL API
    
    GET /outlets?query={query}
    """
    
    try:
        response = requests.get(
            f"{api_server_url}/outlets",
            params={"query": query},
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            
            return {
                "success": True,
                "outlets": data["results"],
                "response": data["response"],
                "sql": data["sql"],
                "count": data["count"],
                "error_type": None
            }
        
        elif response.status_code == 400:
            error_data = response.json()
            print(f"   - Validation error")
            
            return {
                "success": False,
                "error_type": "validation_error",
                "error_message": error_data.get("detail", "Invalid query")
            }
        
        else:
            print(f"   - Status: {response.status_code}")
            
            return {
                "success": False,
                "error_type": "server_error",
                "error_message": f"API returned {response.status_code}"
            }
    
    except Timeout:
        print(f"   - Request timeout")
        
        return {
            "success": False,
            "error_type": "timeout",
            "error_message": "Outlets API timeout"
        }
    
    except ConnectionError:
        print(f"   - Connection failed")
        
        return {
            "success": False,
            "error_type": "connection_error",
            "error_message": "Cannot connect to Outlets API"
        }
    
    except Exception as e:
        print(f"   - Error: {e}")
        
        return {
            "success": False,
            "error_type": "unknown_error",
            "error_message": str(e)
        }


# STEP 5: Executes Planned Actions
def execute_action(state: AgentState) -> AgentState:
    """Execute the planned action"""
    
    action_type = state.get("action_type", "finish")
    
    action_log = {
        "action": action_type,
        "timestamp": datetime.now().isoformat(),
        "reason": state.get("last_decision_reason")
    }
    
    if action_type == "call_calculator":
        return execute_calculator(state, action_log)
    elif action_type == "call_products":
        return execute_products(state, action_log)
    elif action_type == "call_outlets":
        return execute_outlets(state, action_log)
    elif action_type == "chat":
        return execute_chat(state, action_log)
    else:
        return execute_finish(state, action_log)

def execute_calculator(state: AgentState, action_log: Dict) -> AgentState:
    """
    Execute calculator with API call and error handling.
    
    This is the ENHANCED version for Part 3 that:
    1. Extracts expression from user message
    2. Calls real Calculator API
    3. Handles API errors gracefully
    4. Falls back to local calculation if API fails
    5. Generates user-friendly error messages
    """
    
    last_message = state["messages"][-1].content
    
    # Extract expression using LLM
    try:
        extraction_prompt = f"""Extract ONLY the mathematical expression from: "{last_message}"
Return just the expression (e.g., "5+5", "10*2") with no other text."""
        
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        expression = response.content.strip()
        
    except Exception as e:
        expression = last_message  # Fallback
    
    # Call Calculator API
    result = call_calculator_api(expression)
    
    # Generate response based on result
    if result["success"]:
        # Success - show result
        if result.get("used_fallback"):
            response_content = f"The result of {expression} is {result['result']}. (Note: Calculated locally as API was unavailable)"
        else:
            response_content = f"The result of {expression} is {result['result']}."
        
        response_msg = AIMessage(content=response_content)
        
    else:
        # Error - explain what went wrong
        error_type = result.get("error_type", "unknown")
        error_msg = result.get("error_message", "Unknown error")
        
        if error_type == "validation_error":
            response_content = f"Sorry, the expression '{expression}' has an invalid format. Please use only numbers and operators (+, -, *, /)."
        elif error_type == "server_error":
            response_content = f"I'm having trouble connecting to the calculator service. Please try again in a moment."
        else:
            response_content = f"Sorry, I couldn't calculate that: {error_msg}"
        
        response_msg = AIMessage(content=response_content)
    
    # Log the action with full error details
    action_log["tool"] = "calculator_api"
    action_log["result"] = {
        "success": result["success"],
        "result": result.get("result"),
        "error_type": result.get("error_type"),
        "status_code": result.get("status_code"),
        "used_fallback": result.get("used_fallback", False)
    }
    
    # Track API errors
    api_errors = []
    if not result["success"]:
        api_errors = [{
            "tool": "calculator",
            "expression": expression,
            "error_type": result["error_type"],
            "error_message": result["error_message"],
            "timestamp": datetime.now().isoformat()
        }]
    
    return {
        "messages": [response_msg],
        "action_history": [action_log],
        "tool_calls": ["calculator_api"],
        "action_type": "finish",
        "api_errors": api_errors,
    }

def execute_products(state: AgentState, action_log: Dict) -> AgentState:
    """Execute products RAG search"""
    
    last_message = state["messages"][-1].content
    
    result = call_products_api(last_message, top_k=3)
    
    if result["success"]:
        # Use AI response if available, otherwise fall back to rule-based summary
        response_content = result.get("ai_response") or result.get("summary", "Found products matching your search.")
        response_msg = AIMessage(content=response_content)
    else:
        if result["error_type"] == "connection_error":
            response_msg = AIMessage(content="Sorry, I'm having trouble accessing the product catalog right now. Please try again later.")
        else:
            response_msg = AIMessage(content="Sorry, I couldn't find any products matching your search.")
    
    action_log["tool"] = "products_rag_api"
    action_log["result"] = {"success": result["success"], "count": result.get("count", 0)}
    
    return {
        "messages": [response_msg],
        "action_history": [action_log],
        "tool_calls": ["products_rag_api"],
        "action_type": "finish"
    }

def execute_outlets(state: AgentState, action_log: Dict) -> AgentState:
    """Execute outlets Text2SQL query"""

    last_message = state["messages"][-1].content
    location = state.get("location")
    outlet_name = state.get("outlet_name")

    print(f"\n🏪 EXECUTE_OUTLETS - Input State:")
    print(f"   Location: {location}")
    print(f"   Outlet Name: {outlet_name}")
    print(f"   User Message: {last_message}")

    query = last_message

    # If we have context and message is ambiguous, enhance it
    if (location or outlet_name) and len(last_message.split()) < 6:
        context_parts = []
        if outlet_name:
            context_parts.append(f"outlet: {outlet_name}")
        if location:
            context_parts.append(f"location: {location}")

        if context_parts:
            query = f"{last_message} ({', '.join(context_parts)})"

    # Call the API with the query
    result = call_outlets_api(query)
    print(f"   - Outlets API returned {result}")

    # Update action_log with tool information first
    action_log["tool"] = "outlets_text2sql_api"
    action_log["result"] = {"success": result["success"], "count": result.get("count", 0)}

    # Initialize updated state values
    new_outlet_name = outlet_name
    new_location = location

    if result["success"]:
        # Use the formatted response from the API
        response_msg = AIMessage(content=result["response"])

        # Update state with outlet info if found
        if result.get("outlets") and len(result["outlets"]) > 0:
            first_outlet = result["outlets"][0]
            new_outlet_name = first_outlet.get("name")
            new_location = first_outlet.get("location")
    else:
        # Handle errors
        if result["error_type"] == "connection_error":
            response_msg = AIMessage(content="Sorry, I'm having trouble accessing outlet information right now. Please try again later.")
        else:
            response_msg = AIMessage(content="Sorry, I couldn't find any outlets matching your query.")

    print(f"   📤 EXECUTE_OUTLETS - Output State:")
    print(f"   New Outlet Name: {new_outlet_name}")
    print(f"   New Location: {new_location}")

    return {
        "messages": [response_msg],
        "action_history": [action_log],
        "tool_calls": ["outlets_text2sql_api"],
        "action_type": "finish",
        "outlet_name": new_outlet_name,
        "location": new_location
    }

def execute_chat(state: AgentState, action_log: Dict) -> AgentState:
    """
    Execute general chat - friendly conversation without calling external APIs.
    
    This handles:
    - Greetings (hi, hello, hey)
    - Thank you messages
    - General questions about the service
    - Chitchat
    """
    last_message = state["messages"][-1].content
    conversation_history = state.get("messages", [])
    
    # Build context from recent conversation
    recent_context = ""
    if len(conversation_history) > 1:
        # Get last 3 exchanges for context
        recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history[:-1]
        recent_context = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in recent_messages
        ])
    
    # Create chat prompt
    chat_prompt = f"""You are a friendly ZUS Coffee chatbot assistant. 

**Recent Conversation:**
{recent_context if recent_context else "No prior conversation"}

**User's Current Message:** "{last_message}"

**Your Capabilities:**
- Help users find ZUS Coffee outlets
- Search for products (tumblers, mugs, drinkware)
- Calculate numbers
- Provide information about ZUS Coffee

**Instructions:**
- Be warm, friendly, and helpful
- Keep responses concise (2-3 sentences max)
- If user greets you, greet back warmly
- If user thanks you, acknowledge graciously
- If user asks what you can do, briefly list your capabilities
- If user asks general questions, answer helpfully
- Use markdown **bold** for emphasis where appropriate

Generate a natural, friendly response:"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a helpful, friendly customer service chatbot for ZUS Coffee."),
            HumanMessage(content=chat_prompt)
        ])
        
        response_content = response.content.strip()
        response_msg = AIMessage(content=response_content)
        
    except Exception as e:
        print(f"   - Chat LLM error: {e}")
        # Fallback responses
        last_message_lower = last_message.lower()
        
        if any(greeting in last_message_lower for greeting in ["hi", "hello", "hey"]):
            response_content = "Hello! 👋 I'm the ZUS Coffee assistant. I can help you find outlets, search for products, or calculate numbers. How can I help you today?"
        elif any(thanks in last_message_lower for thanks in ["thank", "thanks"]):
            response_content = "You're welcome! Feel free to ask if you need anything else. ☕"
        elif "what can you do" in last_message_lower or "help" in last_message_lower:
            response_content = "I can help you:\n- **Find ZUS Coffee outlets** and their details\n- **Search for products** like tumblers and drinkware\n- **Calculate numbers** for you\n\nWhat would you like to know?"
        else:
            response_content = "I'm here to help! You can ask me about ZUS Coffee outlets, products, or calculations. What would you like to know?"
        
        response_msg = AIMessage(content=response_content)
    
    action_log["tool"] = "chat_llm"
    action_log["result"] = {"success": True}
    
    return {
        "messages": [response_msg],
        "action_history": [action_log],
        "tool_calls": ["chat_llm"],
        "action_type": "finish"
    }

def execute_finish(state: AgentState, action_log: Dict) -> AgentState:
    """Finish - no more actions needed."""
    response_msg = AIMessage(content="Is there anything else I can help you with?")
    
    action_log["tool"] = "none"
    
    return {
        "messages": [response_msg],
        "action_history": [action_log],
        "action_type": "finish"
    }


# STEP 6: Build Agent Graph
def create_agent_graph():
    """
    Creates the agentic planning graph.
    
    Flow:
    1. Extract slots from user input
    2. Plan next action (THE CONTROLLER)
    3. Execute planned action
    4. Return to user (END)

    Graph:
        IN -> extract_slots -> plan_action -> execute_action -> OUT
    """
    workflow = StateGraph(AgentState)
    
    # Define nodes
    workflow.add_node("extract_slots", extract_slots)
    workflow.add_node("plan_action", plan_next_action)
    workflow.add_node("execute_action", execute_action)
    
    # Define flow
    workflow.set_entry_point("extract_slots")
    workflow.add_edge("extract_slots", "plan_action")
    workflow.add_edge("plan_action", "execute_action")
    workflow.add_edge("execute_action", END)
    
    # Add memory
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)


# STEP 7: FastAPI Server Setup
app = FastAPI(title="Agentic Chat API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    message: str
    thread_id: str
    metadata: Dict[str, Any] = {}

# Store agent graphs per thread (in-memory for now)
agent_graphs = {}
thread_states = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that frontend calls.

    Receives user message and thread_id, processes through agent graph,
    returns bot response with metadata.
    """
    try:
        thread_id = request.thread_id

        # Get or create agent graph for this thread
        if thread_id not in agent_graphs:
            agent_graphs[thread_id] = create_agent_graph()
            thread_states[thread_id] = {
                "messages": [],
                "location": None,
                "outlet_name": None,
                "query_type": None,
                "action_type": None,
                "action_history": [],
                "tool_calls": [],
                "pending_action": None,
                "last_decision_reason": None,
                "api_errors": [],
            }

        app_graph = agent_graphs[thread_id]
        state = thread_states[thread_id]
        config = {"configurable": {"thread_id": thread_id}}

        # Add user message to state
        state["messages"].append(HumanMessage(content=request.message))

        # Run agent graph
        result = app_graph.invoke(state, config)

        # Update thread state
        thread_states[thread_id] = result

        print(f"\n💾 FINAL STATE SAVED:")
        print(f"   Location: {result.get('location')}")
        print(f"   Outlet Name: {result.get('outlet_name')}")

        # Extract bot response
        bot_message = result["messages"][-1].content

        # Build metadata for frontend
        metadata = {}

        if result.get("action_history"):
            last_action = result["action_history"][-1]
            metadata["thinking"] = {
                "notice": last_action.get("reason", "Processing your request"),
                "computing": f"Action: {last_action['action']}",
                "preparing": f"Tool: {last_action.get('tool', 'none')}"
            }
            metadata["tool"] = last_action.get("tool")

        # Include error info if any
        if result.get("api_errors"):
            metadata["errors"] = result["api_errors"]

        return ChatResponse(
            message=bot_message,
            thread_id=thread_id,
            metadata=metadata
        )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agentic-chat-api"}



if __name__ == "__main__":
    # Check if API key is set
    if api_key_openai is None:
        print(" > WARNING: Please set your OPENAI_API_KEY environment variable!")
        print("\n > For now, the bot will use fallback keyword matching.\n")

    uvicorn.run(app, host="0.0.0.0", port=8111, log_level="info")