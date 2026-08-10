from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()
llm=ChatOpenAI(
    model="Qwen/Qwen3-4B-Instruct-2507",
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
    temperature=0,
    streaming=True
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    message= state["messages"]
    response=llm.invoke(message)

    return {"messages": [response]}


checkpointer = InMemorySaver()

mygraph=StateGraph(ChatState)
mygraph.add_node("chat_node", chat_node)

mygraph.add_edge(START, "chat_node")
mygraph.add_edge("chat_node", END)

chatbot=mygraph.compile(checkpointer=checkpointer)