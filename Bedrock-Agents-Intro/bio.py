# sample payload for agent: My name is Ram Vegiraju, what is twice my age?

# mock data for agent to query, in real-world scenario replace this with a database
biodata = {"Name": "Ram Vegiraju", "Age": 24}

def getAge(payload):
    print("--------------")
    print(payload)
    print("--------------")
    name = payload['parameters'][0]['value']
    if name not in biodata["Name"]:
        raise ValueError(f"This name: {name} is not in our database, please enter a valid name")
    return biodata["Age"]

def lambda_handler(event, context):
    print("--------------")
    print(event)
    print("--------------")
    action = event['actionGroup']
    api_path = event['apiPath']
    print("--------------")
    print(action)
    print(api_path)
    print("--------------")

    if api_path == '/biodata/{name}/age':
        body = getAge(event)
    else:
        body = {"{}::{} is not a valid api, try another one.".format(action, api_path)}
    
    response_body = {
        'application/json': {
            'body': body
        }
    }

    action_response = {
        'actionGroup': event['actionGroup'],
        'apiPath': event['apiPath'],
        'httpMethod': event['httpMethod'],
        'httpStatusCode': 200,
        'responseBody': response_body
    }

    mock_api_response = {'response': action_response}
    return mock_api_response