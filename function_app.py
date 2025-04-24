import azure.functions as func
import logging
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from openai import AsyncAzureOpenAI, AzureOpenAI, OpenAI
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# Load environment variables
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]

print(f"AZURE_SEARCH_ADMIN_KEY: {search_endpoint}")

search_admin_key = os.environ["AZURE_SEARCH_ADMIN_KEY"]

print(f"AZURE_SEARCH_ADMIN_KEY: {search_admin_key}")

index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
azure_openai_chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

# Initialize Azure Search and OpenAI clients
credential = AzureKeyCredential(search_admin_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)
client = AsyncAzureOpenAI(
api_version=azure_openai_api_version,
azure_endpoint=azure_openai_endpoint,
api_key=azure_openai_key,
)

async def extract_titles(query: str) -> list:
    """
    Extract titles from a user query using OpenAI.
    Args:
        query (str): The text query from which titles should be extracted.
    Returns:
        list: A list of extracted titles.
    """
    response = await client.chat.completions.create(
        model=azure_openai_chat_deployment,
        messages=[
            {"role": "system", "content": "Extract titles from the query. List of titles extracted from the query. Complete file names are considered titles. If there are no titles in the query, provide an empty list. For example, in the query 'Find the report on sales and the summary of the meeting using 'myreport.pdf', the titles would be ['myreport.pdf']. If no titles are found, return an empty list."},
            {"role": "user", "content": f"Extract the titles from this query: {query}"}
        ]
    )
    # Extract titles from OpenAI response
    titles = response.choices[0].message.get("content", "").split(", ")
    return titles


@app.route(route="handle_query_handler")
async def handle_query_handler(req: func.HttpRequest) -> func.HttpResponse:
    """
    Handles requests to process a query, extract titles, fetch relevant content,
    and generate an answer using OpenAI.
    Input:
        HTTP POST request with body:
        {
            "query": "Your search query"
        }
    Output:
        HTTP Response:
        - Status 200: Returns a JSON object containing the OpenAI-generated response and relevant content.
        - Status 400: If no query is provided.
        - Status 500: If an error occurs during processing.
    """
    logging.info('QueryHandler function processed a request.')
    try:
        # Parse request body
        body = req.get_json()
        query = body.get("query", "")
        if not query:
            return func.HttpResponse("No query provided.", status_code=400)
        # Extract titles using OpenAI
        titles = await extract_titles(query)
        # Search content by titles if any are extracted
        formatted_results = ""
        if titles:
            filter_query = " or ".join([f"title eq '{title}'" for title in titles])
            results = await search_client.search(filter=filter_query, select=["title", "content"])
            formatted_results = "\n".join(
                [f"Title: {result['title']}\nContent: {result['content']}" async for result in results]
            )
            if not formatted_results:
                formatted_results = "No relevant content found."
        # Use OpenAI to answer the query based on the retrieved content
        response = await client.chat.completions.create(
            model=azure_openai_chat_deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers queries. You do not have access to the internet, but you can use documents in the chat history to answer the question. If the documents do not contain the answer, say 'I don't know'. You must cite your answer with the titles of the documents used. If you are unsure, say 'I don't know'."},
                {"role": "user", "content": f"Answer this query: '{query}' using the following documents:\n{formatted_results}"}
            ]
        )
        openai_answer = response.choices[0].message.get("content", "No answer generated.")
        # Response object with the OpenAI answer and retrieved content
        response_data = {
            "openai_answer": openai_answer,
            "retrieved_content": formatted_results
        }
        return func.HttpResponse(json.dumps(response_data), status_code=200, mimetype="application/json")
    except Exception as e:
        logging.error(f"Error in QueryHandler: {str(e)}")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)