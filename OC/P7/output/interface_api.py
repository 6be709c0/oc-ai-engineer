import requests
import json

### TO NOTE: THE LAMBDA MAY TAKE UP TO 1 MIN TO START
### MEANING THAT YOU CAN GET A "REQUEST TIMED OUT TESTING LOCALLY ON THE FIRST ATTEMPT"
### PREFER TO USE https://dev.sentiment.parf.ai/ DIRECTLY

print("TO NOTE: THE LAMBDA MAY TAKE UP TO 1 MIN TO START")
print("MEANING THAT YOU CAN GET A \"REQUEST TIMED OUT TESTING LOCALLY ON THE FIRST ATTEMPT\"")
print("PREFER TO USE https://dev.sentiment.parf.ai/ DIRECTLY\"")

print("\n\nTesting Sentiment API")
### TEST SENTIMENT
url = "https://dev.sentiment.parf.ai/api/sentiment"

payload = json.dumps({
  "text": "I am so sad, this is very bad news, terrible!"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

print("\n\nTesting Posting feedback API")

### TEST SUBMIT FEEDBACK
url = "https://dev.sentiment.parf.ai/api/sentiment/feedback"

payload = json.dumps({
  "text": "I am so sad, this is very bad news, terrible!",
  "sentiment": 0.88,
  "feedback": False
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

## TEST GET FEEDBACK
print("\n\nTesting Getting Feedback API")
import requests

url = "https://dev.sentiment.parf.ai/api/sentiment/feedback"

payload = ""
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
