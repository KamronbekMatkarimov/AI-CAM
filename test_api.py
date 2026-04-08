import requests
import json
import time

# Test API with different services
test_cases = [
    {"service": "default", "key": "abc123", "expected": 200},
    {"service": "service1", "key": "key456", "expected": 200},
    {"service": "service2", "key": "key789", "expected": 200},
    {"service": "invalid", "key": "wrong", "expected": 401},
]

for test in test_cases:
    print(f"\nTesting service: {test['service']}")
    try:
        files = {'file': open('test_images/360_F_294670928_YKNwAmem2OGY86CmgvLFux0oCZvlKYFi.jpg', 'rb')}
        data = {'metadata': json.dumps({'service': test['service'], 'camera_name': 'test'})}
        headers = {'X-API-Key': test['key']}

        start = time.time()
        response = requests.post('http://localhost:5000/api/v1/tasks/submit', files=files, data=data, headers=headers)
        end = time.time()

        print(f"Status: {response.status_code} (expected: {test['expected']})")
        print(f"Time: {end - start:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            print(f"People count: {result.get('count', 'N/A')}")
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Exception: {e}")