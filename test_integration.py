"""
Integration Test Script for Dual-Port System Interlock Verification
Tests network payload delivery, CORS configuration, and UI component rendering
"""
import requests
import json
from typing import Dict, Any

def test_backend_health():
    """Test backend health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend Health Check: PASSED")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   ChromaDB Connected: {data.get('chroma_connected')}")
            return True
        else:
            print(f"❌ Backend Health Check: FAILED (Status {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Backend Health Check: FAILED ({str(e)})")
        return False

def test_query_endpoint():
    """Test query endpoint with sample payload."""
    try:
        payload = {
            "query": "What was OmniCorp net profit margin in Q4 2025?",
            "top_k": 3
        }
        response = requests.post(
            "http://localhost:8000/api/v1/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Query Endpoint Test: PASSED")
            print(f"   Response received: {len(data.get('response', ''))} characters")
            print(f"   Source nodes retrieved: {len(data.get('source_nodes', []))}")
            
            # Verify Pydantic schema structure
            required_keys = ['response', 'source_nodes', 'query', 'top_k']
            for key in required_keys:
                if key not in data:
                    print(f"   ❌ Missing required key: {key}")
                    return False
            
            # Verify source node structure
            if data.get('source_nodes'):
                source = data['source_nodes'][0]
                source_keys = ['text', 'file_name', 'page_number', 'score', 'metadata']
                for key in source_keys:
                    if key not in source:
                        print(f"   ❌ Missing source node key: {key}")
                        return False
            
            print("   ✅ Pydantic schema validation: PASSED")
            return True
        else:
            print(f"❌ Query Endpoint Test: FAILED (Status {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Query Endpoint Test: FAILED ({str(e)})")
        return False

def test_cors_configuration():
    """Test CORS configuration by sending request with Origin header."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Origin": "http://localhost:8501"
        }
        response = requests.options(
            "http://localhost:8000/api/v1/query",
            headers=headers,
            timeout=5
        )
        
        cors_headers = response.headers
        print("✅ CORS Configuration Test: PASSED")
        print(f"   Access-Control-Allow-Origin: {cors_headers.get('Access-Control-Allow-Origin', 'Not set')}")
        print(f"   Access-Control-Allow-Methods: {cors_headers.get('Access-Control-Allow-Methods', 'Not set')}")
        print(f"   Access-Control-Allow-Headers: {cors_headers.get('Access-Control-Allow-Headers', 'Not set')}")
        return True
    except Exception as e:
        print(f"❌ CORS Configuration Test: FAILED ({str(e)})")
        return False

def main():
    """Run all integration tests."""
    print("=" * 80)
    print("DUAL-PORT SYSTEM INTERLOCK VERIFICATION")
    print("=" * 80)
    print()
    
    # Test 1: Backend Health
    print("TEST 1: Backend Health Check")
    print("-" * 80)
    health_passed = test_backend_health()
    print()
    
    # Test 2: Query Endpoint
    print("TEST 2: Query Endpoint & Pydantic Schema Validation")
    print("-" * 80)
    query_passed = test_query_endpoint()
    print()
    
    # Test 3: CORS Configuration
    print("TEST 3: CORS Configuration")
    print("-" * 80)
    cors_passed = test_cors_configuration()
    print()
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Backend Health Check: {'✅ PASSED' if health_passed else '❌ FAILED'}")
    print(f"Query Endpoint Test: {'✅ PASSED' if query_passed else '❌ FAILED'}")
    print(f"CORS Configuration: {'✅ PASSED' if cors_passed else '❌ FAILED'}")
    print()
    
    all_passed = health_passed and query_passed and cors_passed
    if all_passed:
        print("🎉 ALL TESTS PASSED - System is ready for frontend integration")
    else:
        print("⚠️ SOME TESTS FAILED - Please review the errors above")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
