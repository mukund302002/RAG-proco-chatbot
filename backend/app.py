from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import get_embeddings, get_response, split_text_into_chunks,split_text_and_embeddings
import os
from sqlalchemy import create_engine
from supabase import create_client, Client
from utils import extract_text_from_pdf
from dotenv import load_dotenv
import logging
import json
import torch
from scipy.spatial.distance import cosine


load_dotenv()

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)
DATABASE_URL = os.getenv('POSTGRES_URL')
engine = create_engine(DATABASE_URL)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
print(SUPABASE_KEY)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files :
            return jsonify({'error': 'No file part'}), 400
        
        if 'id' not in request.form:
            return jsonify({'error': "id missing"}), 400

        file = request.files['file']
        custom_id=request.form['id']


        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        text = extract_text_from_pdf(file)

        chunks = split_text_into_chunks(text,200)

        embedding = get_embeddings(chunks)

        # Convert the tensor to a list
        embedding_list = embedding.tolist()
        data = {'id': custom_id,'content':chunks, 'embeddings': embedding_list}

        # Insert data into Supabase
        response = supabase_client.table('pdfs').insert(data).execute()
        # logging.debug(f"Supabase response: {response}")


        if 'error' in response:  
            logging.error(f"Error response from Supabase: {response}")
            return jsonify({'error': 'Failed to upload PDF'}), 500
        else:
            pdf_id = response.data[0]['id']
            return jsonify({'id': pdf_id}), 201
    except Exception as e:
        logging.error(f"Error uploading PDF: {e}")
        return jsonify({'error': str(e)}), 500
    






api_token="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"



@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        logging.debug(f"content of data: {data}")
        query_text = data['query']
        pdf_id = data['pdf_id']
        

        # Retrieve all rows with the given PDF ID
        response = supabase_client.table('pdfs').select('embeddings', 'content').eq('id', pdf_id).execute()
        
        if response.data:
            # Concatenate the embeddings and content from all retrieved rows
            concatenated_embeddings = []
            concatenated_content = []

            for row in response.data:
                concatenated_embeddings.extend(row['embeddings'])
                concatenated_content.extend(row['content'])  # Assuming row['content'] is a list of chunks

            concated_embeddings=torch.tensor(concatenated_embeddings)
            

            def compute_cosine_similarity(embedding1, embedding2):
                return 1 - cosine(embedding1, embedding2)
            
            def find_top_n_chunks(query, all_embeddings, all_chunks, top_n=3):
                query_embedding = get_embeddings(query)
    
                similarities = []
                for i, embedding in enumerate(all_embeddings):
                    similarity = compute_cosine_similarity(query_embedding, embedding)
                    similarities.append((similarity, all_chunks[i]))
    
                # Sort by similarity in descending order
                sorted_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)
    
                # Get top N chunks
                top_chunks = sorted_chunks[:top_n]
    
                return top_chunks
            top_chunks = find_top_n_chunks(query_text, concatenated_embeddings, concatenated_content, top_n=3)
            logging.debug("top chunks are: {top_chunks}")



            query_embeddings = get_embeddings(query_text)
            logging.debug("Generated query embeddings")

            # Get response from the model
            response_text = get_response(top_chunks,query_text)
            logging.debug(f"Response text: {response_text}")

            return jsonify({'response': response_text})
        else:
            logging.error(f"Error response from Supabase: {response}")
            return jsonify({'error': 'PDF not found'}), 404
    except Exception as e:
        logging.error(f"Error querying: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
