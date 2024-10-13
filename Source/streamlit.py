import streamlit as st
from typing import Sequence
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from data_process import retriever  # Import your retriever module
from Set_env import llm  # Import your LLM setup

# Define prompts
contextualize_q_system_prompt = (
    "Dựa trên lịch sử cuộc trò chuyện và câu hỏi mới nhất của người dùng có thể tham chiếu đến ngữ cảnh trong lịch sử trò chuyện, "
    "hãy tạo thành một câu hỏi độc lập có thể hiểu được mà không cần lịch sử cuộc trò chuyện."
     " KHÔNG trả lời câu hỏi, chỉ cần điều chỉnh lại nếu cần, nếu không thì giữ nguyên."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "Bạn là một trợ lý cho các nhiệm vụ trả lời câu hỏi một cách chi tiết. Sử dụng những mẩu ngữ cảnh được truy xuất sau để trả lời câu hỏi. "
    "Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Define the state schema
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

# Define a function to call the model
def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

# Set up the workflow
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Initialize memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Streamlit app
st.title("Chatbot Interface")
st.write("Ask your questions below:")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.markdown(f"**You:** {message.content}")
        elif isinstance(message, AIMessage):
            st.markdown(f"**Chatbot:** {message.content}")

# Create a text input for user questions
user_input = st.text_input("Your question:", "")

if st.button("Submit"):
    if user_input:
        # Invoke the chatbot model
        config = {"configurable": {"thread_id": "abc123"}}
        result = app.invoke({"input": user_input}, config=config)

        # Update chat history in session state
        st.session_state.chat_history.append(HumanMessage(user_input))
        st.session_state.chat_history.append(AIMessage(result["answer"]))

        # Display the answer
        st.write("### Answer:")
        st.write(result["answer"])
        
    else:
        st.warning("Please enter a question.")
