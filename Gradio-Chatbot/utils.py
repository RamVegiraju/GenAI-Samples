from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

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

# model
modelId = "anthropic.claude-v2"
llm = Bedrock(
    model_id=modelId,
    model_kwargs={"max_tokens_to_sample": 1000}
)
