from typing_extensions import TypedDict
import operator
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from nodes import *
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()

llm = ChatOllama(model="qwen2:0.5b", temperature=0) 


class State(TypedDict):
    question: Annotated[list, operator.add]
    pre_answers: Annotated[str, operator.add]
    context: Annotated[list, operator.add]
    final_output: str



builder_tavily = StateGraph(State)
builder_tavily.add_node("search_tavily",search_tavily)
builder_tavily.add_node("generate_summary",generate_summary)
builder_tavily.add_edge(START, "search_tavily")
builder_tavily.add_edge("search_tavily", "generate_summary")
builder_tavily.add_edge("generate_summary", END)
subgraph_tavily = builder_tavily.compile()


builder_duck_duck_go = StateGraph(State)
builder_duck_duck_go.add_node("search_duck_duck_go",search_duck_duck_go)
builder_duck_duck_go.add_node("generate_summary",generate_summary)
builder_duck_duck_go.add_edge(START, "search_duck_duck_go")
builder_duck_duck_go.add_edge("search_duck_duck_go", "generate_summary")
builder_duck_duck_go.add_edge("generate_summary", END)
subgraph_builder_duck_duck_go = builder_duck_duck_go.compile()


builder_final_answer = StateGraph(State)
builder_final_answer.add_node("get_final_answer",final_answer)
builder_final_answer.add_edge(START, "get_final_answer")
builder_final_answer.add_edge("get_final_answer", END)
builder_final_answer = builder_final_answer.compile()


entry_builder = StateGraph(State)
entry_builder.add_node("Tavily", subgraph_tavily)
entry_builder.add_node("Duck Duck Go", subgraph_builder_duck_duck_go)
entry_builder.add_node("builder_final_answer", builder_final_answer)
entry_builder.add_edge(START, "Tavily")
entry_builder.add_edge(START, "Duck Duck Go")
entry_builder.add_edge("Tavily", "builder_final_answer")
entry_builder.add_edge("Duck Duck Go", "builder_final_answer")
entry_builder.add_edge("builder_final_answer", END)
graph = entry_builder.compile()



png_bytes = graph.get_graph(xray=1).draw_mermaid_png()

# Save the PNG data to a file
with open("research_project_graph.png", "wb") as f:
    f.write(png_bytes)
    


print(graph.invoke({"question":["Explain me what's a linear regression and add the formula"]})['final_output'].content)