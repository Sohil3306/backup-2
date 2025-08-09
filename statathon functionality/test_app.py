import requests
import json

def test_app():
    base_url = "http://127.0.0.1:5000"
    
    # Test 1: Check if the main page loads
    try:
        response = requests.get(base_url)
        print(f"Main page status: {response.status_code}")
        if response.status_code == 200:
            print("✓ Main page loads successfully")
        else:
            print("✗ Main page failed to load")
    except Exception as e:
        print(f"✗ Error accessing main page: {e}")
    
    # Test 2: Check if columns endpoint works
    try:
        response = requests.get(f"{base_url}/get_columns")
        print(f"Columns endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Columns endpoint works: {data}")
        else:
            print("✗ Columns endpoint failed")
    except Exception as e:
        print(f"✗ Error accessing columns endpoint: {e}")

if __name__ == "__main__":
    test_app() 