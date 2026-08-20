
# Backend
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command

from dotenv import load_dotenv
import requests

load_dotenv()


from pydantic import SecretStr

hf_token = os.getenv("HF_TOKEN")
llm = ChatOpenAI(
    model="Qwen/Qwen3-4B-Instruct-2507",
    base_url="https://router.huggingface.co/v1",
    api_key=SecretStr(hf_token) if hf_token is not None else None,
    temperature=0,
)


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g., AAPL, GOOGL).
    using Alpha Vantage with API key in the URL"""

    url= f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=O20TPEI7SM08V7SF"
    r = requests.get(url)
    return r.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    Human-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision{"yes" / anything else}.
    """

    # This will pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}.")

    if isinstance(decision, str) and decision.lower() == "yes":
        return{
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }

    else:
        return{
            "status": "cancelled",
            "message": f"purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }




tools = [get_stock_price,purchase_stock]
llm_with_tools = llm.bind_tools(tools)


# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]




# Nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""

    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# Checkpointer
memory = MemorySaver()



# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)




# Simple CLI
if __name__ == "__main__":

    #use a fixed thread_id so the conversation is persisted in memory
    thread_id = "demo-thread"

    while True:
        user_input = input("YOU: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Build the state for this turn
        state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph (may hit an interrupt)
        result = chatbot.invoke(state,config={"configurable": {"thread_id": thread_id}})


        # Check for HITL interrupt from purchase_stock
        interrupts = result.get("__interrupt__",[])

        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            # Resume graph with the human decision ("yes" / "no" / whatever)
            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}},
            )


        # Get the latest message from the assistant
        messages = result["messages"]
        last_msz = messages[-1]
        print(f"Bot: {last_msz.content}\n")