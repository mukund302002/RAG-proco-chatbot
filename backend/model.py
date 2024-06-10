from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
from sentence_transformers import util
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import requests
import time

def split_text_and_embeddings(words, embeddings, max_chunk_size):
    """
    Split the text and embeddings into chunks.
    
    Args:
    words (list): The list of words to be split into chunks.
    embeddings (list): The list of embeddings to be split into corresponding chunks.
    max_chunk_size (int): The maximum size of each chunk.
    
    Returns:
    list of str: List of text chunks.
    list of list: List of corresponding embeddings for each chunk.
    """
    chunks = []
    chunk_embeddings = []
    current_chunk = []
    current_embeddings = []

    for idx, word in enumerate(words):
        current_chunk.append(word)
        current_embeddings.append(embeddings[idx])
        
        if len(" ".join(current_chunk)) >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            chunk_embeddings.append(current_embeddings)
            current_chunk = []
            current_embeddings = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        chunk_embeddings.append(current_embeddings)

    return chunks, chunk_embeddings

def get_embeddings(text):
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model = SentenceTransformer(embed_model_name)
    embeddings = embed_model.encode(text, convert_to_tensor=True)
    return embeddings



logging.basicConfig(level=logging.DEBUG)
api_token="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"


def get_response(top_chunks,query_text):
    # # Find the most similar passages
    # logging.debug("started the response function")

    # # Ensure the embeddings and passages are correctly paired
    # # if len(passages) != len(passage_embeddings):
    # #     raise ValueError("Number of passages and passage embeddings must match.")

    # # Compute similarity scores
    # similarity_scores = util.pytorch_cos_sim(query_embeddings, passage_embeddings)
    # top_scores, top_indices = torch.topk(similarity_scores, k=3)

    # # Retrieve the top chunks based on similarity scores
    # top_chunks = [passages[idx.item()] for idx in top_indices[0]]
    # logging.debug(f"Top chunks: {top_chunks}")

    # # Concatenate the top chunks to form the input passages
    input_passages = top_chunks
    # Prepare the input text for the model
    input_text = f"Context: {input_passages}\n\nQuestion: {query_text}"
    logging.debug(f"Input text for the model: {input_text}")
    
    # Define headers for the Hugging Face Inference API request
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # Define the payload for the request
    payload = {
        "inputs": input_text,
        "parameters": {
            "max_length": 1024,
            "num_return_sequences": 1,
        }
    }
    logging.debug("Headers and payload prepared")

    # Make the request to the Hugging Face Inference API
    logging.debug("Making request to Hugging Face Inference API")
    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        headers=headers,
        json=payload
    )

    # Check for errors in the response
    if response.status_code != 200:
        logging.error(f"Error querying Hugging Face API: {response.status_code}, {response.text}")
        raise ValueError(f"Error querying Hugging Face API: {response.status_code}, {response.text}")

    logging.debug("Request successful")
    res = response.json()
    logging.debug(f"Response: {res}")

    # Extract the generated text from the response
    generated_text = res[0]["summary_text"]

    return generated_text








from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text_into_chunks(text, max_chunk_size):
    """
    Recursively split text into chunks of a maximum specified size.
    
    Args:
    text (str): The input text to be split into chunks.
    max_chunk_size (int): The maximum size of each chunk.
    
    Returns:
    list of str: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

    
   