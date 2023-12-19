import boto3
import time
from time import gmtime, strftime

client = boto3.client('bedrock-agent')
agent_name= "bio-agent" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

response = client.create_agent(
    agentName = agent_name,
    
)