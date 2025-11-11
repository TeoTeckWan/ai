from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
import operator
import os
import json

load_dotenv()
api_key_openai = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-5-nano-2025-08-07", 
    temperature=0,  # Deterministic for intent parsing
    api_key=api_key_openai
)

# STEP 1: Define the State Schema
class ConversationState(TypedDict):
    """
    State schema that tracks all information across conversation turns.
    
    Attributes:
        messages: Full conversation history (user + bot messages)
        location: Extracted location from user (e.g., "Petaling Jaya")
        outlet_name: Specific outlet name (e.g., "SS 2")
        query_type: Intent / What the user wants (e.g., "opening_time", "check_outlet")
        context: Additional context for decision making
    """
    messages:    Annotated[List[BaseMessage], operator.add]
    location:    Optional[str]
    outlet_name: Optional[str]
    query_type:  Optional[str]
    context:     dict


# STEP 2: Mock Data 
OUTLETS_DATA = {
    "Petaling Jaya": {
        "SS 2": {
            "opening_time": "9:00 AM",
            "closing_time": "10:00 PM",
            "address": "Jalan SS 2/24, SS 2, 47300 Petaling Jaya"
        },
        "Damansara Uptown": {
            "opening_time": "8:00 AM",
            "closing_time": "11:00 PM",
            "address": "Jalan SS 21/37, Damansara Uptown, 47400 Petaling Jaya"
        }
    },
    "Kuala Lumpur": {
        "KLCC": {
            "opening_time": "10:00 AM",
            "closing_time": "10:00 PM",
            "address": "Suria KLCC, Kuala Lumpur City Centre"
        }
    }
}


# STEP 3: LLM-Based Intent & Slot Extraction
def extract_intent_and_slots_with_llm(user_message: str, current_state: ConversationState) -> dict:
    """
    Use LLM to extract intent and entities from user message.
    
    Args:
        user_message: What the user just said
        current_state: Current conversation state for context
    
    Returns:
        dict with extracted intent and slots
    """
    # Get conversation history for context
    recent_messages = current_state.get("messages", [])[-6:]  # Last 6 messages
    conversation_context = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}"
        for m in recent_messages
    ])
    
    # Get current state for context
    current_location = current_state.get("location")
    current_outlet = current_state.get("outlet_name")
    
    # Create extraction prompt
    system_prompt = """You are an intent classifier for a coffee shop chatbot (ZUS Coffee).

Your job is to extract:
1. **intent**: What the user wants
   - "check_outlet": Looking for outlet existence/information
   - "opening_time": Asking about opening hours
   - "closing_time": Asking about closing hours
   - "address": Asking for outlet address
   - "unclear": Cannot determine intent

2. **location**: Area/city mentioned
   - "Petaling Jaya" (also: PJ, Petaling)
   - "Kuala Lumpur" (also: KL, KLCC area)
   - Keep existing if not mentioned in new message

3. **outlet_name**: Specific outlet name
   - "SS 2" (also: SS2)
   - "Damansara Uptown" (also: Uptown)
   - "KLCC"
   - Keep existing if not mentioned in new message

**Context Awareness:**
- If location/outlet was mentioned earlier, keep it unless user changes it
- Handle pronouns like "there", "it", "that one" by using context

**IMPORTANT**: Return ONLY valid JSON, no other text.
"""

    user_prompt = f"""Current conversation:
{conversation_context}

Current state:
- Location: {current_location or "Unknown"}
- Outlet: {current_outlet or "Unknown"}

New user message: "{user_message}"

Extract intent and entities. Return ONLY this JSON format:
{{"intent": "...", "location": "...", "outlet_name": "..."}}

Use null for unknown values. Keep existing values if not mentioned in new message."""

    try:
        # Call OpenAI
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Parse JSON response
        result = json.loads(response.content)
        
        # Preserve existing values if LLM returns null
        return {
            "intent": result.get("intent"),
            "location": result.get("location") or current_state.get("location"),
            "outlet_name": result.get("outlet_name") or current_state.get("outlet_name")
        }
    
    except json.JSONDecodeError as e:
        print(f" > LLM returned invalid JSON: {response.content}")
        print(f" > Error: {e}")
        # Fallback to simple keyword matching
        return extract_intent_and_slots_fallback(user_message, current_state)
    
    except Exception as e:
        print(f" > Error calling LLM: {e}")
        # Fallback to simple keyword matching
        return extract_intent_and_slots_fallback(user_message, current_state)

def extract_intent_and_slots_fallback(user_message: str, current_state: ConversationState) -> dict:
    """
    Fallback method using simple keyword matching if LLM fails.
    This ensures the bot never crashes due to LLM errors.
    """
    message_lower = user_message.lower()
    
    # Extract intent
    intent = None
    if any(word in message_lower for word in ["opening", "open", "hours", "time"]):
        intent = "opening_time"
    elif any(word in message_lower for word in ["closing", "close"]):
        intent = "closing_time"
    elif any(word in message_lower for word in ["address", "where"]):
        intent = "address"
    elif any(word in message_lower for word in ["outlet", "branch", "store", "location"]):
        intent = "check_outlet"
    
    # Extract location
    location = current_state.get("location")
    if "petaling jaya" in message_lower or "pj" in message_lower:
        location = "Petaling Jaya"
    elif "kuala lumpur" in message_lower or "kl" in message_lower:
        location = "Kuala Lumpur"
    
    # Extract outlet name
    outlet_name = current_state.get("outlet_name")
    if "ss 2" in message_lower or "ss2" in message_lower:
        outlet_name = "SS 2"
    elif "damansara uptown" in message_lower or "uptown" in message_lower:
        outlet_name = "Damansara Uptown"
    elif "klcc" in message_lower:
        outlet_name = "KLCC"
    
    return {
        "intent": intent,
        "location": location,
        "outlet_name": outlet_name
    }


# STEP 4: Node Functions (Actions the Bot Can Take)
def process_user_input(state: ConversationState) -> ConversationState:
    """
    Node 1: Process the latest user message and extract information using LLM.
    """
    last_message = state["messages"][-1]
    user_text = last_message.content
    
    # Extract intent and slots using OpenAI
    extracted = extract_intent_and_slots_with_llm(user_text, state)
    
    print(f" |=> Extracted - Location: {extracted['location']}, Outlet: {extracted['outlet_name']}, Intent: {extracted['intent']}")
    
    # Update state with extracted information
    return {
        "query_type": extracted["intent"] or state.get("query_type"),
        "location": extracted["location"] or state.get("location"),
        "outlet_name": extracted["outlet_name"] or state.get("outlet_name"),
        "context": state.get("context", {})
    }

def ask_location(state: ConversationState) -> ConversationState:
    """
    Node 2: Bot asks for location if missing.
    Use LLM to generate natural response.
    """
    # Use LLM to generate a natural response
    system_prompt = "You are a friendly ZUS Coffee chatbot. Generate a brief, natural response asking for location."
    user_prompt = "The user hasn't told us their location yet. Ask them which area they're interested in (Petaling Jaya or Kuala Lumpur)."
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = response.content
    except Exception as e:
        print(f" > LLM error, using fallback response: {e}")
        content = "Which area are you looking for? For example, Petaling Jaya or Kuala Lumpur?"
    
    return {"messages": [AIMessage(content=content)]}

def ask_outlet(state: ConversationState) -> ConversationState:
    """
    Node 3: Bot asks for specific outlet name.
    Use LLM to generate natural response with available outlets.
    """
    location = state.get("location")
    
    # Get available outlets in that location
    if location and location in OUTLETS_DATA:
        outlets = list(OUTLETS_DATA[location].keys())
        outlets_text = ", ".join(outlets)
        
        # Use LLM to generate natural response
        system_prompt = "You are a friendly ZUS Coffee chatbot. Generate a brief, natural response."
        user_prompt = f"The user is asking about {location}. We have these outlets: {outlets_text}. Ask which one they're interested in."
        
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            content = response.content
        except Exception as e:
            print(f" > LLM error, using fallback response: {e}")
            content = f"Yes! We have outlets in {location}. Which one are you asking about? ({outlets_text})"
    else:
        content = "Which specific outlet are you referring to?"
    
    return {"messages": [AIMessage(content=content)]}

def answer_query(state: ConversationState) -> ConversationState:
    """
    Node 4: Bot has all info needed - provide the answer using LLM for natural response.
    """
    location = state["location"]
    outlet_name = state["outlet_name"]
    query_type = state["query_type"]
    
    # Look up the information from our data
    try:
        outlet_data = OUTLETS_DATA[location][outlet_name]
        
        # Prepare data for LLM
        if query_type == "opening_time":
            info = outlet_data['opening_time']
            query = "opening time"
        elif query_type == "closing_time":
            info = outlet_data['closing_time']
            query = "closing time"
        elif query_type == "address":
            info = outlet_data['address']
            query = "address"
        elif query_type == "check_outlet":
            info = f"Open from {outlet_data['opening_time']} to {outlet_data['closing_time']} at {outlet_data['address']}"
            query = "outlet information"
        else:
            info = str(outlet_data)
            query = "information"
        
        # Use LLM to generate natural response
        system_prompt = "You are a friendly ZUS Coffee chatbot. Generate a natural, helpful response with the provided information."
        user_prompt = f"User asked about the {query} for {outlet_name} outlet in {location}. The {query} is: {info}. Provide a friendly response."
        
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            answer = response.content
        except Exception as e:
            print(f" > LLM error, using fallback response: {e}")
            # Fallback responses
            if query_type == "opening_time":
                answer = f"The {outlet_name} outlet in {location} opens at {outlet_data['opening_time']}."
            elif query_type == "closing_time":
                answer = f"The {outlet_name} outlet closes at {outlet_data['closing_time']}."
            elif query_type == "address":
                answer = f"The {outlet_name} outlet is located at {outlet_data['address']}."
            else:
                answer = f"Here's info about {outlet_name}: {outlet_data}"
        
        response_msg = AIMessage(content=answer)
    
    except KeyError:
        response_msg = AIMessage(content=f"Sorry, I don't have information about {outlet_name} in {location}.")
    
    return {"messages": [response_msg]}

def handle_unclear(state: ConversationState) -> ConversationState:
    """
    Node 5: Bot couldn't understand - ask for clarification using LLM.
    """
    # Use LLM to generate natural clarification request
    system_prompt = "You are a friendly ZUS Coffee chatbot. Generate a brief, natural response asking for clarification."
    user_prompt = "The user's intent is unclear. Politely ask them to clarify if they want outlet locations, opening hours, or other information."
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = response.content
    except Exception as e:
        print(f" > LLM error, using fallback response: {e}")
        content = "I'm not sure what you're asking. Are you looking for outlet locations or operating hours?"
    
    return {"messages": [AIMessage(content=content)]}


# STEP 5: Decision Logic - What Should Bot Do Next?
def determine_next_action(state: ConversationState) -> str:
    """
    Decision tree:
    1. If we have all info needed -> answer_query
    2. If missing outlet name but have location -> ask_outlet
    3. If missing location -> ask_location
    4. Otherwise -> handle_unclear
    """
    intent = state.get("query_type")
    location = state.get("location")
    outlet_name = state.get("outlet_name")
    
    # If we have intent and all required info, answer the query
    if intent and location and outlet_name:
        return "answer_query"
    
    # If we know location but not specific outlet
    if location and not outlet_name:
        return "ask_outlet"
    
    # If we don't know location yet
    if not location:
        return "ask_location"
    
    # Unclear intent
    return "handle_unclear"


# STEP 6: Build the Conversation Graph
def create_conversation_graph():
    """
    Creates the LangGraph state machine for conversation flow.

    Graph:
                                                -> ask_location   -> OUT
        IN -> process_input -> Conditional Gate -> ask_outlet     -> OUT
                                                -> answer_query   -> OUT
                                                -> handle_unclear -> OUT
    """
    workflow = StateGraph(ConversationState)
    
    # Add all possible nodes
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("ask_location", ask_location)
    workflow.add_node("ask_outlet", ask_outlet)
    workflow.add_node("answer_query", answer_query)
    workflow.add_node("handle_unclear", handle_unclear)
    
    # Set entry point
    workflow.set_entry_point("process_input")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "process_input",
        determine_next_action,
        {
            "ask_location": "ask_location",
            "ask_outlet": "ask_outlet",
            "answer_query": "answer_query",
            "handle_unclear": "handle_unclear"
        }
    )
    
    # All response nodes lead to END
    workflow.add_edge("ask_location", END)
    workflow.add_edge("ask_outlet", END)
    workflow.add_edge("answer_query", END)
    workflow.add_edge("handle_unclear", END)
    
    # Add memory
    memory = MemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=memory)
    
    return app


# STEP 7: Conversation Runner
def run_conversation(user_inputs: List[str], thread_id: str = "conversation_1"):
    """
    Run a multi-turn conversation with LLM-enhanced understanding.
    """
    app = create_conversation_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    current_state = {
        "messages": [],
        "location": None,
        "outlet_name": None,
        "query_type": None,
        "context": {}
    }
    
    print("=" * 70)
    print("  -- ZUS COFFEE CHATBOT --")
    print("=" * 70)
    
    for i, user_input in enumerate(user_inputs, 1):
        print(f"\n User (Turn {i}): {user_input}")
        
        current_state["messages"].append(HumanMessage(content=user_input))
        
        # Run the graph
        result = app.invoke(current_state, config)
        current_state = result
        
        # Get bot's response
        bot_response = result["messages"][-1].content
        print(f" Bot: {bot_response}")
        
        # Show current state
        print(f"\n >> [State] Location: {result.get('location')}, "
              f"Outlet: {result.get('outlet_name')}, "
              f"Intent: {result.get('query_type')}")
    
    print("\n" + "=" * 70)
    print(" -- CONVERSATION END --")
    print("=" * 70)
    
    return current_state


# STEP 8: Main Execution
if __name__ == "__main__":
    # Check if API key is set
    if api_key_openai is None:
        print(" > WARNING: Please set your OPENAI_API_KEY environment variable!")
        print("\n > For now, the bot will use fallback keyword matching.\n")
    
    # Test Case 1: Happy Path
    print("\n TEST CASE 1: HAPPY PATH (Assessment Example)")
    happy_path = [
        "Is there an outlet in Petaling Jaya?",
        "SS 2, what's the opening time?",
    ]
    run_conversation(happy_path, thread_id="test_1")
    
    # Test Case 2: Interrupted Path
    print("\n\n TEST CASE 2: INTERRUPTED PATH (Gradual Information)")
    interrupted_path = [
        "What time do you open?",
        "I mean in Petaling Jaya",
        "The SS 2 one"
    ]
    run_conversation(interrupted_path, thread_id="test_2")
    
    # Test Case 3: All info at once
    print("\n\n TEST CASE 3: ALL INFO AT ONCE\n")
    all_at_once = [
        "What time does the SS 2 outlet in Petaling Jaya open?"
    ]
    run_conversation(all_at_once, thread_id="test_3")
    
    # Test Case 4: Natural language variations
    print("\n\n TEST CASE 4: NATURAL LANGUAGE (LLM Advantage)")
    natural_language = [
        "Hey, do you guys have a place in PJ?",
        "Yeah, the one at SS2, when does it close?",
    ]
    run_conversation(natural_language, thread_id="test_4")