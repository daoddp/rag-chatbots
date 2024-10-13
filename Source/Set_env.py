import os

# Set environment variable for LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Set LangChain API key if not already set
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = "your-langchain-api-key"

# Set OpenAI API key if not already set
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "your-open-ai-key"

# Import LangChain's OpenAI interface and initialize the model
from langchain_openai import ChatOpenAI

# Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini")
