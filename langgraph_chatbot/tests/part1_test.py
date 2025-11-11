import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# For import the part1.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from part1 import (
    ConversationState,
    extract_intent_and_slots_with_llm,
    extract_intent_and_slots_fallback,
    determine_next_action,
    create_conversation_graph,
    process_user_input,
    ask_location,
    ask_outlet,
    answer_query,
    handle_unclear,
    api_key_openai
)
from langchain_core.messages import HumanMessage, AIMessage


# Setup Test Data
@pytest.fixture
def empty_state():
    """Fresh conversation state"""
    return {
        "messages": [],
        "location": None,
        "outlet_name": None,
        "query_type": None,
        "context": {}
    }


@pytest.fixture
def partial_state():
    """State with location but no outlet"""
    return {
        "messages": [
            HumanMessage(content="Is there an outlet in Petaling Jaya?")
        ],
        "location": "Petaling Jaya",
        "outlet_name": None,
        "query_type": "check_outlet",
        "context": {}
    }


@pytest.fixture
def complete_state():
    """State with all information"""
    return {
        "messages": [
            HumanMessage(content="Is there an outlet in Petaling Jaya?"),
            AIMessage(content="Yes! Which outlet?"),
            HumanMessage(content="SS 2, what's the opening time?")
        ],
        "location": "Petaling Jaya",
        "outlet_name": "SS 2",
        "query_type": "opening_time",
        "context": {}
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    def _mock_response(intent="check_outlet", location="Petaling Jaya", outlet_name=None):
        mock = Mock()
        
        # Properly format None as JSON null
        location_str = f'"{location}"' if location is not None else "null"
        outlet_str = f'"{outlet_name}"' if outlet_name is not None else "null"
        
        mock.content = f'{{""location": {location_str}, "outlet_name": {outlet_str}, intent": "{intent}"}}'
        return mock
    
    return _mock_response


# Test Fallback Extraction (No LLM)
def test_fallback_extract_location_petaling_jaya():
    """Test fallback location extraction"""
    result = extract_intent_and_slots_fallback(
        "Is there an outlet in Petaling Jaya?",
        {"location": None}
    )
    assert result["location"] == "Petaling Jaya"
    assert result["intent"] == "check_outlet"


def test_fallback_extract_location_abbreviation():
    """Test fallback location extraction with abbreviation"""
    result = extract_intent_and_slots_fallback(
        "Any outlet in PJ?",
        {"location": None}
    )
    assert result["location"] == "Petaling Jaya"


def test_fallback_extract_outlet_name():
    """Test fallback outlet name extraction"""
    result = extract_intent_and_slots_fallback(
        "SS 2 outlet opening time",
        {"location": "Petaling Jaya"}
    )
    assert result["outlet_name"] == "SS 2"
    assert result["intent"] == "opening_time"


def test_fallback_preserve_existing_slots():
    """Test that fallback preserves existing slot values"""
    current_state = {"location": "Petaling Jaya", "outlet_name": "SS 2"}
    result = extract_intent_and_slots_fallback("What time?", current_state)
    
    assert result["location"] == "Petaling Jaya"
    assert result["outlet_name"] == "SS 2"



# Test LLM Extraction (Mock)
@patch('part1.llm')
def test_llm_extract_intent_and_location(mock_llm, empty_state, mock_llm_response):
    """Test LLM-based extraction with mock"""
    mock_llm.invoke.return_value = mock_llm_response(
        intent="check_outlet",
        location="Petaling Jaya",
        outlet_name=None
    )
    
    result = extract_intent_and_slots_with_llm(
        "Is there an outlet in Petaling Jaya?",
        empty_state
    )
    
    assert result["intent"] == "check_outlet"
    assert result["location"] == "Petaling Jaya"
    assert mock_llm.invoke.called


@patch('part1.llm')
def test_llm_extract_with_context(mock_llm, partial_state, mock_llm_response):
    """Test that LLM uses context to preserve location"""
    mock_llm.invoke.return_value = mock_llm_response(
        intent="opening_time",
        location="Petaling Jaya", 
        outlet_name="SS 2"
    )
    
    result = extract_intent_and_slots_with_llm("SS 2", partial_state)
    
    assert result["location"] == "Petaling Jaya"
    assert result["outlet_name"] == "SS 2"


@patch('part1.llm')
def test_llm_extraction_fallback_on_error(mock_llm, empty_state):
    """Test that extraction falls back to keyword matching on LLM error"""
    mock_llm.invoke.side_effect = Exception("API Error")
    
    result = extract_intent_and_slots_with_llm(
        "Is there an outlet in Petaling Jaya?",
        empty_state
    )
    
    # Should still return result using fallback
    assert result["location"] == "Petaling Jaya"
    assert result["intent"] == "check_outlet"


@patch('part1.llm')
def test_llm_extraction_handles_invalid_json(mock_llm, empty_state):
    """Test handling of invalid JSON from LLM"""
    mock_response = Mock()
    mock_response.content = "This is not JSON"
    mock_llm.invoke.return_value = mock_response
    
    result = extract_intent_and_slots_with_llm(
        "Is there an outlet in Petaling Jaya?",
        empty_state
    )
    
    # Should fallback to keyword matching
    assert result["location"] == "Petaling Jaya"



# Test Decision Logic (Same as before)
def test_decision_missing_all_info(empty_state):
    """When no info available, should ask for location"""
    action = determine_next_action(empty_state)
    assert action == "ask_location"


def test_decision_has_location_needs_outlet(partial_state):
    """When has location but no outlet, should ask for outlet"""
    action = determine_next_action(partial_state)
    assert action == "ask_outlet"


def test_decision_has_all_info(complete_state):
    """When has all info, should answer query"""
    action = determine_next_action(complete_state)
    assert action == "answer_query"


def test_decision_unclear_intent(partial_state):
    """When all info is present but intent is missing, should handle as unclear."""
    # This state has location and outlet, but no intent
    unclear_state = partial_state.copy()
    unclear_state["outlet_name"] = "SS 2"
    unclear_state["query_type"] = None # No intent
    
    action = determine_next_action(unclear_state)
    assert action == "handle_unclear"


# Test Node Functions (Mocked LLM calls)
@patch('part1.llm')
def test_process_user_input_with_llm(mock_llm, empty_state, mock_llm_response):
    """Test process_user_input node with mocked LLM"""
    empty_state["messages"] = [
        HumanMessage(content="Is there an outlet in Petaling Jaya?")
    ]
    
    mock_llm.invoke.return_value = mock_llm_response(
        intent="check_outlet",
        location="Petaling Jaya",
        outlet_name=None
    )
    
    result = process_user_input(empty_state)
    
    assert result["location"] == "Petaling Jaya"
    assert result["query_type"] == "check_outlet"


@patch('part1.llm')
def test_ask_location_uses_llm(mock_llm, empty_state):
    """Test that ask_location uses LLM for response generation"""
    mock_response = Mock()
    mock_response.content = "Which area would you like to know about?"
    mock_llm.invoke.return_value = mock_response
    
    result = ask_location(empty_state)
    
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert mock_llm.invoke.called


@patch('part1.llm')
def test_ask_outlet_uses_llm(mock_llm, partial_state):
    """Test that ask_outlet uses LLM for response generation"""
    mock_response = Mock()
    mock_response.content = "We have several outlets in Petaling Jaya. Which one interests you?"
    mock_llm.invoke.return_value = mock_response
    
    result = ask_outlet(partial_state)
    
    assert "messages" in result
    assert mock_llm.invoke.called


@patch('part1.llm')
def test_answer_query_uses_llm(mock_llm, complete_state):
    """Test that answer_query uses LLM for response generation"""
    mock_response = Mock()
    mock_response.content = "The SS 2 outlet opens at 9:00 AM!"
    mock_llm.invoke.return_value = mock_response
    
    result = answer_query(complete_state)
    
    assert "messages" in result
    bot_message = result["messages"][0].content
    assert "9:00 AM" in bot_message or "SS 2" in bot_message
    assert mock_llm.invoke.called


def test_answer_query_handles_invalid_outlet():
    """Test handling of invalid outlet query"""
    state = {
        "messages": [],
        "location": "Petaling Jaya",
        "outlet_name": "NonExistent",
        "query_type": "opening_time",
        "context": {}
    }
    
    result = answer_query(state)
    bot_message = result["messages"][0].content
    assert "sorry" in bot_message.lower() or "don't have" in bot_message.lower()


@patch('part1.llm')
def test_handle_unclear_uses_llm(mock_llm, empty_state):
    """Test that handle_unclear uses LLM"""
    mock_response = Mock()
    mock_response.content = "Could you clarify what you're looking for?"
    mock_llm.invoke.return_value = mock_response
    
    result = handle_unclear(empty_state)
    
    assert "messages" in result
    assert mock_llm.invoke.called


# Integration Tests (Full Conversation Flows)
@patch('part1.llm')
def test_happy_path_full_conversation(mock_llm, mock_llm_response):
    """Test the exact example from assessment with mocked LLM"""
    app = create_conversation_graph()
    config = {"configurable": {"thread_id": "test_happy"}}
    
    # Mock LLM responses for each turn
    mock_llm.invoke.side_effect = [
        # Turn 1: Extract intent from "Is there an outlet in Petaling Jaya?"
        mock_llm_response(intent="check_outlet", location="Petaling Jaya", outlet_name=None),
        # Turn 1: Generate ask_outlet response
        Mock(content="Yes! We have outlets in Petaling Jaya. Which one? (SS 2, Damansara Uptown)"),

        # Turn 2: Extract from "SS 2, what's the opening time?"
        mock_llm_response(intent="opening_time", location="Petaling Jaya", outlet_name="SS 2"),
        # Turn 2: Generate answer
        Mock(content="The SS 2 outlet in Petaling Jaya opens at 9:00 AM!"),
    ]
    
    # Turn 1
    state = {
        "messages": [HumanMessage(content="Is there an outlet in Petaling Jaya?")],
        "location": None,
        "outlet_name": None,
        "query_type": None,
        "context": {}
    }
    result = app.invoke(state, config)
    
    assert result["location"] == "Petaling Jaya"
    
    # Turn 2
    state = result.copy()
    state["messages"].append(HumanMessage(content="SS 2, what's the opening time?"))
    result = app.invoke(state, config)
    
    assert result["outlet_name"] == "SS 2"
    assert result["query_type"] == "opening_time"


@patch('part1.llm')
def test_interrupted_path_gradual_info(mock_llm, mock_llm_response):
    """Test conversation where user provides info gradually"""
    app = create_conversation_graph()
    config = {"configurable": {"thread_id": "test_interrupted"}}
    
    # Mock responses for gradual information
    mock_llm.invoke.side_effect = [
        # Turn 1: No location
        mock_llm_response(intent="opening_time", location=None, outlet_name=None),
        Mock(content="Which area would you like to know about?"),

        # Turn 2: Has location
        mock_llm_response(intent="opening_time", location="Petaling Jaya", outlet_name=None),
        Mock(content="Which outlet in Petaling Jaya?"),

        # Turn 3: Has all
        mock_llm_response(intent="opening_time", location="Petaling Jaya", outlet_name="SS 2"),
        Mock(content="Opens at 9:00 AM!"),
    ]
    
    # Turn 1
    state = {
        "messages": [HumanMessage(content="What time do you open?")],
        "location": None,
        "outlet_name": None,
        "query_type": None,
        "context": {}
    }
    result = app.invoke(state, config)
    assert result["location"] is None
    
    # Turn 2
    state = result.copy()
    state["messages"].append(HumanMessage(content="In Petaling Jaya"))
    result = app.invoke(state, config)
    assert result["location"] == "Petaling Jaya"
    
    # Turn 3
    state = result.copy()
    state["messages"].append(HumanMessage(content="The SS 2 one"))
    result = app.invoke(state, config)
    assert result["outlet_name"] == "SS 2"


# Edge Cases
def test_empty_user_input():
    """Test handling of empty messages"""
    state = {
        "messages": [HumanMessage(content="")],
        "location": None,
        "outlet_name": None,
        "query_type": None,
        "context": {}
    }
    
    # Should not crash
    result = extract_intent_and_slots_fallback("", state)
    assert result is not None


@patch('part1.llm')
def test_llm_timeout_fallback(mock_llm, empty_state):
    """Test that system handles LLM timeout gracefully"""
    mock_llm.invoke.side_effect = TimeoutError("Request timeout")
    
    empty_state["messages"] = [HumanMessage(content="Is there an outlet in PJ?")]
    
    # Should not crash, should use fallback
    result = process_user_input(empty_state)
    assert result is not None
    assert result["location"] == "Petaling Jaya"


def test_case_insensitive_extraction():
    """Test case insensitive slot extraction in fallback"""
    result = extract_intent_and_slots_fallback(
        "WHAT TIME DOES SS 2 IN PETALING JAYA OPEN?",
        {}
    )
    assert result["location"] == "Petaling Jaya"
    assert result["outlet_name"] == "SS 2"


# Run Tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])