from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.retrievers import ArxivRetriever
from services import db_service


def search_arxiv(state):
    """
    Performs a search on arXiv and retrieves search results.

    Args:
        state (dict): A dictionary containing the key 'question' with a list 
                      where the first element is the search query.

    Returns:
        dict: A dictionary containing the key 'context' with the retrieved documents.
    """
    try:
        retriever = ArxivRetriever(load_max_docs=2, get_ful_documents=True)
        docs = retriever.invoke(state['question'][0])
        result = {"context": [docs.page_content]}
        return result
    except:
        return {"context": []}

def read_question(state):
    """
    Reads the question from the state and puts it into the result.
    
    Args:
        state (dict): A dictionary containing the key 'question' with a list 
                      where the first element is the question to be answered.

    Returns:
        dict: A dictionary containing the key 'question' with the question to be answered.
    """
    result = {"question": state['question']}
    return state

def search_tavily(state):
    """
    Performs a search using Tavily and retrieves search results.

    Args:
        state (dict): A dictionary containing the key 'question' with a list 
                      where the first element is the search query.

    Returns:
        dict: A dictionary containing the key 'context' with the search results.
    """
    try:
        tavily_search = TavilySearchResults(max_results=3)
        search_docs = tavily_search.invoke(state['question'][0])
        formatted_search_docs = "\n\n---\n\n".join(
            [f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>' for doc in search_docs]
        )
        result = {"context": [formatted_search_docs]}
        return result
    except:
        return {"context": []}

def search_wikipedia(state):
    """
    Performs a search using Wikipedia and retrieves search results.

    Args:
        state (dict): A dictionary containing the key 'question' with a list 
                      where the first element is the search query.

    Returns:
        dict: A dictionary containing the key 'context' with a list of two strings, 
              where each string contains the content of each search result.
    """
    try:
        search_docs = WikipediaLoader(query=state['question'][0], load_max_docs=2).load()
        formatted_search_docs = "\n\n---\n\n".join(
            [f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>' for doc in search_docs]
        )
        result = {"context": [formatted_search_docs]}
        return result
    except:
        return {"context": []}

def search_duck_duck_go(state):
    """
    Performs a search using DuckDuckGo and retrieves search results.

    Args:
        state (dict): A dictionary containing the key 'question' with a list 
                      where the first element is the search query.

    Returns:
        dict: A dictionary containing the key 'context' with the search results.
    """
    try:
        search = DuckDuckGoSearchRun()
        search_docs = search.invoke(state['question'][0])
        result = {"context": [search_docs]}
        return result
    except:
        return {"context": []}

def generate_summary(state):
    """
    Generates a summary of the given question using the context provided.

    Args:
        state (dict): A dictionary containing the following keys:
            - 'context' (str or list): The context to use when generating the summary.
            - 'question' (list): A list of strings where the first element is the question to be answered.

    Returns:
        dict: A dictionary containing the key 'pre_answers' with the generated summary as its value.
    """
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
    """
    Generates a final and unified answer to a given question using pre-existing answers.

    Args:
        state (dict): A dictionary containing the following keys:
            - 'pre_answers' (str): Pre-existing answers that will be used to form the final answer.
            - 'question' (list): A list where the first element is the question that needs to be answered.

    Returns:
        dict: A dictionary containing the key 'final_output' with the generated final answer as its value.
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    pre_answers = state['pre_answers']
    question = state['question'][0]
    answer_template = "Write a final and unified answer to the question {question} following this pre-answers {pre_answers}"
    answer_instructions = answer_template.format(pre_answers=pre_answers, question=question)
    final_answer_result = llm.invoke(answer_instructions)
    final_answer_text = final_answer_result.content if hasattr(final_answer_result, "content") else str(final_answer_result)
    result = {"final_output": final_answer_text}
    return result
