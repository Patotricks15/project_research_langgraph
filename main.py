from typing_extensions import TypedDict
import operator
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from nodes import *
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0) 


class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]


builder = StateGraph(State)

builder.add_node("search_tavily",search_tavily)
builder.add_node("search_duck_duck_go",search_duck_duck_go)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("search_arxiv", search_arxiv)
builder.add_node("generate_summary", generate_summary)

builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_tavily")
builder.add_edge(START, "search_duck_duck_go")
builder.add_edge(START, "search_arxiv")

builder.add_edge("search_wikipedia", "generate_summary")
builder.add_edge("search_duck_duck_go", "generate_summary")
builder.add_edge("search_tavily", "generate_summary")
builder.add_edge("search_arxiv", "generate_summary")

builder.add_edge("generate_summary", END)

graph = builder.compile()


print(graph.invoke({"question":"Explain me what's a linear regression and add the formula"})['answer'].content)
