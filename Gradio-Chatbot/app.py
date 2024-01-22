import random
import gradio as gr
import os
import openai
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

os.environ['OPENAI_API_KEY'] = 'enter key'


# prompt template for claude, adjust for other models
claude_prompt = PromptTemplate.from_template("""

Human: The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. If the AI does not know
the answer to a question, it truthfully says it does not know.

Current conversation:
<conversation_history>
{history}
</conversation_history>

Here is the human's next reply:
<human_reply>
{input}
</human_reply>

Assistant:
""")

modelId = "anthropic.claude-v2"
llm = Bedrock(
    model_id=modelId,
    model_kwargs={"max_tokens_to_sample": 1000}
)
chain = ConversationChain(
    llm=llm, verbose=False, memory=ConversationBufferMemory(), prompt=claude_prompt
)

def random_response(message, history):
    resp = chain.predict(input = message)
    return resp

demo = gr.ChatInterface(random_response)

if __name__ == "__main__":
    demo.launch()
