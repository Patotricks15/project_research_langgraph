from typing_extensions import TypedDict
import operator
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from services.llm_service import *
from services.db_service import *
from dotenv import load_dotenv


selected_service = MongoDBService(database_name="research_agents", collection_name="results")

load_dotenv()

set_db_service(selected_service)

selected_db_service = selected_service


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

builder_wikipedia = StateGraph(State)
builder_wikipedia.add_node("search_wikipedia",search_wikipedia)
builder_wikipedia.add_node("generate_summary",generate_summary)
builder_wikipedia.add_edge(START, "search_wikipedia")
builder_wikipedia.add_edge("search_wikipedia", "generate_summary")
builder_wikipedia.add_edge("generate_summary", END)
subgraph_builder_wikipedia = builder_wikipedia.compile()

builder_final_answer = StateGraph(State)
builder_final_answer.add_node("get_final_answer",final_answer)
builder_final_answer.add_edge(START, "get_final_answer")
builder_final_answer.add_edge("get_final_answer", END)
builder_final_answer = builder_final_answer.compile()


entry_builder = StateGraph(State)
entry_builder.add_node("Tavily", subgraph_tavily)
entry_builder.add_node("Duck Duck Go", subgraph_builder_duck_duck_go)
entry_builder.add_node("Wikipedia", subgraph_builder_wikipedia)
entry_builder.add_node("builder_final_answer", builder_final_answer)
entry_builder.add_edge(START, "Tavily")
entry_builder.add_edge(START, "Duck Duck Go")
entry_builder.add_edge(START, "Wikipedia")
entry_builder.add_edge("Tavily", "builder_final_answer")
entry_builder.add_edge("Duck Duck Go", "builder_final_answer")
entry_builder.add_edge("Wikipedia", "builder_final_answer")
entry_builder.add_edge("builder_final_answer", END)
graph = entry_builder.compile()



png_bytes = graph.get_graph(xray=1).draw_mermaid_png()

# Save the PNG data to a file
with open("research_project_graph.png", "wb") as f:
    f.write(png_bytes)
    

while True:
    question = input("Enter your question: ")

    initial_state = {"question": [question]}

    final_state = graph.invoke(initial_state)
       
    final_document = {
        "question": final_state.get("question")[0],
        "context": final_state.get("context"),
        "pre_answers": final_state.get("pre_answers"),
        "final_output": final_state.get("final_output"),
    }
    
    selected_db_service.insert_document(final_document)
    
    print("Final Output:", final_document["final_output"])

    print(graph.invoke({"question":[question]})['final_output'])