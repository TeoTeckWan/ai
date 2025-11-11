from dotenv import load_dotenv
import os
import pytest
import requests
import json
from typing import Dict, Any


# Configuration
load_dotenv()
api_server_url = os.getenv("BASE_URL")
TIMEOUT = 30


# FIXTURES
@pytest.fixture(scope="session") # Run once for entire test
def api_url():
    """Base URL for API"""
    return api_server_url


@pytest.fixture(scope="session", autouse=True)
def check_server_running(api_url):
    """Check if server is running before tests"""
    try:
        response = requests.get(f"{api_url}/health", timeout=TIMEOUT)
        assert response.status_code == 200, "Server health check failed"
        print(f"\n✅ Server is running at {api_url}\n")
    except Exception as e:
        pytest.fail(f"Server is not running at {api_url}. Error: {e}")


@pytest.fixture
def disable_downtime(api_url):
    """Ensure downtime is disabled before each test"""
    try:
        # Use the reset endpoint to explicitly disable all test simulations
        requests.get(f"{api_url}/test/reset", timeout=TIMEOUT)
    except:
        pass



# TEST CASE 1: MISSING PARAMETERS
class TestMissingParameters:
    """Test Case 1: Missing Parameters"""
    
    def test_calculator_missing_expression(self, api_url, disable_downtime):
        """Calculator without expression should return 422 error"""
        response = requests.post(
            f"{api_url}/calculate",
            json={},
            timeout=TIMEOUT
        )
        
        # Assert status code is 422 (Unprocessable Entity)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    
    
    def test_calculator_empty_expression(self, api_url, disable_downtime):
        """Calculator with empty expression should return 400/422 error"""
        response = requests.post(
            f"{api_url}/calculate",
            json={"expression": ""},
            timeout=TIMEOUT
        )
        
        # Assert error status code
        assert response.status_code in [400, 422], f"Expected 400 or 422, got {response.status_code}"
        
    
    def test_products_missing_query(self, api_url, disable_downtime):
        """Products without query should return 422 error"""
        response = requests.get(f"{api_url}/products", timeout=TIMEOUT)
        
        # Assert status code
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"
        
    
    def test_outlets_missing_query(self, api_url, disable_downtime):
        """Outlets without query should return 422 error"""
        response = requests.get(f"{api_url}/outlets", timeout=TIMEOUT)
        
        # Assert status code
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"
        


# TEST CASE 2: API DOWNTIME
class TestAPIDowntime:
    """Test Case 2: API Downtime - Simulate HTTP 500/503 errors"""
    
    def test_enable_downtime_simulation(self, api_url):
        """Should be able to enable downtime simulation"""
        response = requests.get(
            f"{api_url}/test/simulate-downtime",
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "downtime_simulation" in data, "Response should indicate downtime status"
        
        # Should toggle (either True or False is fine for first call)
        assert isinstance(data["downtime_simulation"], bool), "downtime_simulation should be boolean"
        
    
    def test_calculator_during_downtime(self, api_url):
        """Calculator should return 503 during downtime"""
        # Reset first to ensure clean state
        requests.get(f"{api_url}/test/reset", timeout=TIMEOUT)

        # Enable downtime (toggle once from disabled state)
        toggle_response = requests.get(
            f"{api_url}/test/simulate-downtime",
            timeout=TIMEOUT
        )

        downtime_status = toggle_response.json().get("downtime_simulation", False)
        assert downtime_status == True, "Downtime should be enabled"

        # Try calculator during downtime
        response = requests.post(
            f"{api_url}/calculate",
            json={"expression": "5 + 5"},
            timeout=TIMEOUT
        )

        # Should return 503 Service Unavailable
        assert response.status_code == 503, f"Expected 503 during downtime, got {response.status_code}"

    
    def test_products_during_downtime(self, api_url):
        """Products should return 503 during downtime"""
        # Reset first to ensure clean state
        requests.get(f"{api_url}/test/reset", timeout=TIMEOUT)

        # Enable downtime (toggle once from disabled state)
        toggle_response = requests.get(
            f"{api_url}/test/simulate-downtime",
            timeout=TIMEOUT
        )
        downtime_status = toggle_response.json().get("downtime_simulation", False)
        assert downtime_status == True, "Downtime should be enabled"

        # Try products during downtime
        response = requests.get(
            f"{api_url}/products",
            params={"query": "tumbler"},
            timeout=TIMEOUT
        )

        assert response.status_code == 503, f"Expected 503 during downtime, got {response.status_code}"

    
    def test_outlets_during_downtime(self, api_url):
        """Outlets should return 503 during downtime"""
        # Reset first to ensure clean state
        requests.get(f"{api_url}/test/reset", timeout=TIMEOUT)

        # Enable downtime (toggle once from disabled state)
        toggle_response = requests.get(
            f"{api_url}/test/simulate-downtime",
            timeout=TIMEOUT
        )
        downtime_status = toggle_response.json().get("downtime_simulation", False)
        assert downtime_status == True, "Downtime should be enabled"

        # Try outlets during downtime
        response = requests.get(
            f"{api_url}/outlets",
            params={"query": "SS2 opening time"},
            timeout=TIMEOUT
        )

        assert response.status_code == 503, f"Expected 503 during downtime, got {response.status_code}"

    
    def test_disable_downtime_simulation(self, api_url):
        """Should be able to disable downtime simulation"""
        # Toggle twice to ensure we end up with disabled state
        response1 = requests.get(
            f"{api_url}/test/simulate-downtime",
            timeout=TIMEOUT
        )
        response2 = requests.get(
            f"{api_url}/test/simulate-downtime",
            timeout=TIMEOUT
        )
        
        # After two toggles, check the status
        health_response = requests.get(f"{api_url}/health", timeout=TIMEOUT)
        health_data = health_response.json()
        
        # Downtime should be in a known state
        assert "test_mode" in health_data, "Health response should contain test_mode"


# TEST CASE 3: MALICIOUS PAYLOADS
class TestMaliciousPayloads:
    """Test Case 3: Malicious Payloads - SQL Injection, XSS, Code Injection"""
    
    # SQL Injection Tests
    def test_sql_injection_drop_table(self, api_url, disable_downtime):
        """Should block DROP TABLE SQL injection"""
        response = requests.get(
            f"{api_url}/outlets",
            params={"query": "'; DROP TABLE outlets; --"},
            timeout=TIMEOUT
        )
        
        # Should be blocked (400)
        assert response.status_code == 400, f"SQL injection should be blocked, got {response.status_code}"
        
    
    def test_sql_injection_union_select(self, api_url, disable_downtime):
        """Should block UNION SELECT SQL injection"""
        response = requests.get(
            f"{api_url}/outlets",
            params={"query": "1' UNION SELECT * FROM outlets--"},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 400, f"SQL injection should be blocked, got {response.status_code}"
        
    
    def test_sql_injection_always_true(self, api_url, disable_downtime):
        """Should block OR 1=1 SQL injection"""
        response = requests.get(
            f"{api_url}/outlets",
            params={"query": "1' OR '1'='1"},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 400, f"SQL injection should be blocked, got {response.status_code}"
    
    
    def test_sql_injection_comment(self, api_url, disable_downtime):
        """Should block SQL comment injection"""
        response = requests.get(
            f"{api_url}/outlets",
            params={"query": "admin'--"},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 400, f"SQL injection should be blocked, got {response.status_code}"
    

    def test_sql_injection_encoded(self, api_url, disable_downtime):
        response = requests.get(
            f"{api_url}/outlets",
            params={"query": "%27%3B%20DROP%20TABLE%20outlets%3B--"},
            timeout=TIMEOUT
        )
        assert response.status_code == 400


    def test_sql_injection_mixed_case(self, api_url, disable_downtime):
        response = requests.get(
            f"{api_url}/outlets",
            params={"query": "DrOp TaBlE outlets"},
            timeout=TIMEOUT
        )
        assert response.status_code == 400

    
    # XSS Tests
    def test_xss_script_tag(self, api_url, disable_downtime):
        """Should block XSS with script tag"""
        response = requests.get(
            f"{api_url}/products",
            params={"query": "<script>alert('xss')</script>"},
            timeout=TIMEOUT
        )
        
        # Should be blocked or sanitized
        assert response.status_code in [400, 422], f"XSS should be blocked, got {response.status_code}"
    
    
    def test_xss_event_handler(self, api_url, disable_downtime):
        """Should block XSS with event handler"""
        response = requests.get(
            f"{api_url}/products",
            params={"query": "<img src=x onerror=alert('xss')>"},
            timeout=TIMEOUT
        )
        
        assert response.status_code in [400, 422], f"XSS should be blocked, got {response.status_code}"

    
    def test_xss_javascript_protocol(self, api_url, disable_downtime):
        """Should block XSS with javascript: protocol"""
        response = requests.get(
            f"{api_url}/products",
            params={"query": "javascript:alert('xss')"},
            timeout=TIMEOUT
        )
        
        assert response.status_code in [400, 422], f"XSS should be blocked, got {response.status_code}"
    
    
    # Code Injection Tests
    def test_code_injection_import(self, api_url, disable_downtime):
        """Should block code injection with __import__"""
        response = requests.post(
            f"{api_url}/calculate",
            json={"expression": "__import__('os').system('ls')"},
            timeout=TIMEOUT
        )
        
        # Should be blocked by regex validator
        assert response.status_code in [400, 422], f"Code injection should be blocked, got {response.status_code}"
    

    def test_code_injection_eval(self, api_url, disable_downtime):
        """Should block code injection with eval"""
        response = requests.post(
            f"{api_url}/calculate",
            json={"expression": "eval('1+1')"},
            timeout=TIMEOUT
        )
        
        assert response.status_code in [400, 422], f"Code injection should be blocked, got {response.status_code}"


# TEST CASE 4: COMPREHENSIVE MALICIOUS INPUT TEST
class TestComprehensiveMalicious:
    """Test the /test/malicious endpoint if available"""
    
    def test_sql_injection_comprehensive(self, api_url, disable_downtime):
        """Test comprehensive SQL injection detection"""
        response = requests.post(
            f"{api_url}/test/malicious",
            json={"test_type": "sql_injection"},
            timeout=TIMEOUT
        )
        
        # Should return results
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        
        # Should have test results
        assert "tests" in data, "Should contain test results"
        assert len(data["tests"]) > 0, "Should have run tests"
        
        # All tests should be blocked
        for test in data["tests"]:
            assert test.get("blocked", False) == True, f"Test should be blocked: {test.get('input')}"


if __name__ == "__main__":
    print("\nRun with: pytest test_unhappy_flows_pytest.py -v\n")

    # Run pytest programmatically
    import sys
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])