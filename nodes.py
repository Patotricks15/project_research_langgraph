from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.retrievers import ArxivRetriever

def search_arxiv(state):
    """
    Search arxiv for the question and return the top 2 results.

    Args:
        state (dict): The current state of the graph.

    Returns:
        dict: A dictionary with a single key "context" which contains the formatted search results.
    """
    retriever = ArxivRetriever(
        load_max_docs=2,
        get_ful_documents=True,
    )

    docs = retriever.invoke(state['question'][0])
    return {"context": [docs]}

def read_question(state):
    
    return {"question": state['question']}


def search_tavily(state):
    """
    Search tavily for the question and return the top 3 results.

    Args:
        state (dict): The current state of the graph.

    Returns:
        dict: A dictionary with a single key "context" which contains the formatted search results.
    """
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'][0])
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"context": [formatted_search_docs]}

def search_wikipedia(state):
    """
    Search wikipedia for the question and return the top 2 results.

    Args:
        state (dict): The current state of the graph.

    Returns:
        dict: A dictionary with a single key "context" which contains the formatted search results.
    """
    search_docs = WikipediaLoader(query=state['question'][0], load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def search_duck_duck_go(state):
    """
    Search DuckDuckGo for the question and return the top result.

    Args:
        state (dict): The current state of the graph.

    Returns:
        dict: A dictionary with a single key "context" which contains the formatted search results.
    """
    search = DuckDuckGoSearchRun()
    search_docs = search.invoke(state['question'][0])    
    return {"context": [search_docs]}

def generate_summary(state):
    """
    Generate a summary of the given question using the given context.

    Args:
        state (dict): The current state of the graph.

    Returns:
        dict: A dictionary with a single key "answer" which contains the generated summary.
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    context = state['context']
    question = state['question'][0]
    
    # If context is a list of strings, join them into one string.
    if isinstance(context, list):
        context = " ".join(context)
    
    answer_template = "Answer the question {question} using this context: {context}"
    answer_instructions = answer_template.format(question=question, context=context)
    
    answer = llm.invoke(answer_instructions)
    
    # Extract the content if the answer is an AIMessage object.
    answer_text = answer.content if hasattr(answer, "content") else str(answer)
    
    return {"pre_answers": answer_text}

def final_answer(state):
    """
    Use the pre_answers to generate a final and unified answer to the question.

    Args:
        state (dict): The current state of the graph.

    Returns:
        dict: A dictionary with a single key "final_output" which contains the generated final answer.
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    pre_answers = state['pre_answers']
    question = state['question'][0]
    answer_template = """Write a final and unified answer to the question {question} following this pre-answers {pre_answers}"""
    answer_instructions = answer_template.format(pre_answers=pre_answers, question=question)    
    final_answer = llm.invoke(answer_instructions)
    return {"final_output": final_answer}