from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import ArxivRetriever

def search_arxiv(state):
    retriever = ArxivRetriever(
        load_max_docs=2,
        get_ful_documents=True,
    )

    docs = retriever.invoke(state['question'])
    return {"context": [docs]}


def search_tavily(state):

    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"context": [formatted_search_docs]}

def search_wikipedia(state):
    search_docs = WikipediaLoader(query=state['question'], load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def search_duck_duck_go(state):
    search = DuckDuckGoSearchRun()
    search_docs = search.invoke(state['question'])    
    return {"context": [search_docs]}

def generate_answer(state):
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0) 
    context = state['context']
    question = state['question']
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, 
                                                       context=context)    
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content=f"Answer the question.")])
    return {"answer": answer}