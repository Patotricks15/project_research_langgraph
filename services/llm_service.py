from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.retrievers import ArxivRetriever
from services import db_service


def search_arxiv(state):
    retriever = ArxivRetriever(load_max_docs=2, get_ful_documents=True)
    docs = retriever.invoke(state['question'][0])
    result = {"context": [docs]}
    return result

def read_question(state):
    result = {"question": state['question']}
    return state

def search_tavily(state):
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'][0])
    formatted_search_docs = "\n\n---\n\n".join(
        [f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>' for doc in search_docs]
    )
    result = {"context": [formatted_search_docs]}
    return result

def search_wikipedia(state):
    search_docs = WikipediaLoader(query=state['question'][0], load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>' for doc in search_docs]
    )
    result = {"context": [formatted_search_docs]}
    return result

def search_duck_duck_go(state):
    search = DuckDuckGoSearchRun()
    search_docs = search.invoke(state['question'][0])
    result = {"context": [search_docs]}
    return result

def generate_summary(state):
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    context = state['context']
    question = state['question'][0]
    if isinstance(context, list):
        context = " ".join(context)
    answer_template = "Answer the question {question} using this context: {context}"
    answer_instructions = answer_template.format(question=question, context=context)
    answer = llm.invoke(answer_instructions)
    answer_text = answer.content if hasattr(answer, "content") else str(answer)
    result = {"pre_answers": answer_text}
    return result

def final_answer(state):
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    pre_answers = state['pre_answers']
    question = state['question'][0]
    answer_template = "Write a final and unified answer to the question {question} following this pre-answers {pre_answers}"
    answer_instructions = answer_template.format(pre_answers=pre_answers, question=question)
    final_answer_result = llm.invoke(answer_instructions)
    final_answer_text = final_answer_result.content if hasattr(final_answer_result, "content") else str(final_answer_result)
    result = {"final_output": final_answer_text}
    return result
