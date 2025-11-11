from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from datetime import datetime
from requests.exceptions import RequestException, Timeout, ConnectionError
import os
import json
import operator
import requests


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
    context:     dict
    
    action_type:          Optional[str]  # 'ask', 'call_tool', 'finish'
    action_history:       Annotated[List[Dict], operator.add]  # Log of decisions
    tool_calls:           List[str]  # List of tools used
    pending_action:       Optional[Dict[str, Any]]  # Action to execute
    last_decision_reason: Optional[str]  # Explanation of decision

    api_errors:  List[Dict[str, Any]]  # Track API failures
    retry_count: int  # Number of retries attempted


# STEP 2: Extract Slots 
def extract_slots(state: AgentState) -> AgentState:
    """
    Extract information from user message with outlet data validation.
    """

    last_message = state["messages"][-1].content
    current_location = state.get("location")
    current_outlet = state.get("outlet_name")
    
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
   - null - general questions

**Rules:**
- Preserve existing context unless user explicitly changes it
- Product queries don't need location/outlet

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
        
        # Merge with existing state - preserve context
        new_location = extracted.get("location") or current_location
        new_outlet = extracted.get("outlet_name") or current_outlet
        new_query_type = extracted.get("query_type") or state.get("query_type")
        
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
    1. Check if user wants calculation -> call_calculator
    2. Check if user wants product info -> call_products_rag
    3. Check if user wants outlet info -> check_slots_then_call_outlets
    4. Check if missing info -> ask_followup
    5. Otherwise -> finish
    
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
4. "finish" - Query complete

Important:
- Product queries should use "call_products"
- Outlet queries should use "call_outlets" (no need to ask for location first - API handles it)

Analyze the user's intent and decide which action to take.

Return ONLY valid JSON:
{{"action": "...", "reason": "brief explanation", "missing_info": ["list", "of", "missing", "slots"] or null}}
"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert planning agent."),
            HumanMessage(content=planning_prompt)
        ])
        
        decision = json.loads(response.content)
        action = decision.get("action", "finish")
        reason = decision.get("reason", "No specific reason")
        missing_info = decision.get("missing_info", [])
        
        return {
            "action_type": action,
            "last_decision_reason": reason,
            "pending_action": {
                "type": action,
                "missing_info": missing_info,
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
    
    # Check for calculator intent
    if any(word in last_message for word in ["calculate", "plus", "minus", "+", "-", "*", "/"]):
        return {
            "action_type": "call_calculator",
            "last_decision_reason": "Detected arithmetic intent",
            "pending_action": {"type": "call_calculator"}
        }
    
    # Check for product intent - BOTH keyword match AND query_type
    if query_type == "product_query" or any(word in last_message for word in ["product", "tumbler", "mug", "drinkware", "cup"]):
        return {
            "action_type": "call_products",
            "last_decision_reason": "Detected product query intent",
            "pending_action": {"type": "call_products"}
        }
    
    # Check for outlet intent
    if any(word in last_message for word in ["outlet", "open", "close", "where", "address"]):
        return {
            "action_type": "call_outlets",
            "last_decision_reason": "Outlet query detected",
            "pending_action": {"type": "call_outlets"}
        }
    
    # Default: finish
    return {
        "action_type": "finish",
        "last_decision_reason": "No clear intent or query complete",
        "pending_action": {"type": "finish"}
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
    
    # Handle API failures with fallback
    if not result["success"] and result.get("error_type") in ["connection_error", "timeout"]:
        result = calculate_locally(expression)
    
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
        "retry_count": 0
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
    
    result = call_outlets_api(last_message)
    
    if result["success"]:
        # Use the formatted response from the API
        response_msg = AIMessage(content=result["response"])
    else:
        if result["error_type"] == "connection_error":
            response_msg = AIMessage(content="Sorry, I'm having trouble accessing outlet information right now. Please try again later.")
        else:
            response_msg = AIMessage(content="Sorry, I couldn't find any outlets matching your query.")
    
    action_log["tool"] = "outlets_text2sql_api"
    action_log["result"] = {"success": result["success"], "count": result.get("count", 0)}
    
    return {
        "messages": [response_msg],
        "action_history": [action_log],
        "tool_calls": ["outlets_text2sql_api"],
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


# STEP 7: Run Conversation
def run_agent_conversation(user_inputs: List[str], thread_id: str = "agent_1"):
    """Run a conversation with the agentic planner."""
    app = create_agent_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    state = {
        "messages": [],
        "location": None,
        "outlet_name": None,
        "query_type": None,
        "context": {},
        "action_type": None,
        "action_history": [],
        "tool_calls": [],
        "pending_action": None,
        "last_decision_reason": None,
        "api_errors": [],
        "retry_count": 0
    }
    
    for i, user_input in enumerate(user_inputs, 1):
        print(f"\n🔄 TURN {i}")
        print(f"👤 User: {user_input}")
        
        state["messages"].append(HumanMessage(content=user_input))
        
        # Run agent
        result = app.invoke(state, config)
        state = result
        
        # Print bot response
        bot_response = result["messages"][-1].content
        print(f"🤖 Bot: {bot_response}")
        
        # Show decision summary
        if result.get("action_history"):
            last_action = result["action_history"][-1]
            print(f"\n * Decision Summary:")
            print(f"   Action: {last_action['action']}")
            print(f"   Reason: {last_action['reason']}")
            if last_action.get("tool"):
                print(f"   Tool Used: {last_action['tool']}")

        # Show API errors if any
        if result.get("api_errors"):
            print(f"\n*  API Errors:")
            for error in result["api_errors"]:
                print(f"   - {error['error_type']}: {error['error_message']}")
    


if __name__ == "__main__":  
    # Check if API key is set
    if api_key_openai is None:
        print(" > WARNING: Please set your OPENAI_API_KEY environment variable!")
        print("\n > For now, the bot will use fallback keyword matching.\n")
      
    # Test 1: Calculator
    # print("\n\n🧪 TEST 1: CALCULATOR TOOL")

    # run_agent_conversation([
    #     "What's 25 + 17?"
    # ], thread_id="test_calc")
    
    # # Test 1A: Successful calculation / Off the API Server to test the fallback condition
    # print("\n\n🧪 TEST 1: SUCCESSFUL CALCULATION")
    # run_agent_conversation([
    #     "What's 25 + 17 + a?"
    # ], thread_id="test_success")
    
    # # Test 1B: Complex expression
    # print("\n\n🧪 TEST 2: COMPLEX EXPRESSION")
    # run_agent_conversation([
    #     "Calculate (10 + 5) * 2"
    # ], thread_id="test_complex")
    
    # Test 2: Products RAG
    # print("\n\n🧪 TEST 2: PRODUCTS RAG")
    # run_agent_conversation([
    #     "Show me your tumblers"
    # ], thread_id="test_products")
    
    # # Test 3: Outlets Text2SQL
    # print("\n\n🧪 TEST 3: OUTLETS TEXT2SQL")
    # run_agent_conversation([
    #     "What time does SS 2 open?"
    # ], thread_id="test_outlets")
    
    # Test 4: Mixed queries
    print("\n\n🧪 TEST 4: MIXED QUERIES")
    run_agent_conversation([
        "Calculate 10 * 5",
        "Do you have ceramic mugs?",
        "Where is the Damansara Uptown outlet?"
    ], thread_id="test_mixed")















    # Test 2: Products
    # print("\n\n🧪 TEST 2: PRODUCTS SEARCH")
    # run_agent_conversation([
    #     "Show me your tumblers"
    # ], thread_id="test_products")
    
    # # Test 3: Multi-turn outlets
    # print("\n\n🧪 TEST 3: OUTLETS QUERY (Multi-turn)")
    # run_agent_conversation([
    #     "Is there an outlet in Petaling Jaya?",
    #     "SS 2, what time do you open?"
    # ], thread_id="test_outlets")
    
    # # Test 4: Reverse lookup - user gives outlet first
    # print("\n\n🧪 TEST 4: REVERSE LOOKUP - Outlet mentioned first (no location)")
    # run_agent_conversation([
    #     "What time does SS 2 open?"
    # ], thread_id="test_reverse_lookup")
    
    # # Test 5: Another reverse lookup test
    # print("\n\n🧪 TEST 5: REVERSE LOOKUP - Damansara Uptown address")
    # run_agent_conversation([
    #     "Where is Damansara Uptown outlet?"
    # ], thread_id="test_reverse_lookup_2")
    
    # # Test 6: Mixed tools
    # print("\n\n🧪 TEST 6: MIXED TOOLS")
    # run_agent_conversation([
    #     "What's 10 times 5?",
    #     "Do you have any mugs?",
    #     "Where's your Damansara Uptown outlet?"
    # ], thread_id="test_mixed")