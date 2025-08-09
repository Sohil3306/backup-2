import requests
import os

def test_file_upload():
    """Test the file upload functionality"""
    
    # Test file path
    test_file = "sample_data.csv"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found!")
        return
    
    print(f"✅ Test file found: {test_file}")
    
    # Prepare the upload
    with open(test_file, 'rb') as f:
        files = {'file': (test_file, f, 'text/csv')}
        
        print("📤 Uploading file...")
        
        try:
            response = requests.post('http://127.0.0.1:5000/upload', files=files)
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print("✅ Upload successful!")
                    print(f"📋 Data shape: {data['data_info']['shape']}")
                    print(f"📋 Columns: {data['data_info']['columns']}")
                    return True
                else:
                    print(f"❌ Upload failed: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ HTTP error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to server. Is the Flask app running?")
            return False
        except Exception as e:
            print(f"❌ Error during upload: {e}")
            return False

if __name__ == "__main__":
    print("🧪 Testing File Upload Functionality")
    print("=" * 40)
    
    success = test_file_upload()
    
    if success:
        print("\n🎉 File upload test PASSED!")
        print("The application should now work correctly.")
    else:
        print("\n💥 File upload test FAILED!")
        print("Please check the server logs for more details.") 