from flask import Flask, render_template, request
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from google.cloud import bigquery
from google.oauth2 import service_account

import re

app = Flask(__name__)

# Load the Annoy index and QA data
embedding_dim = 384
index = AnnoyIndex(embedding_dim, 'angular')
index.load(r'D:\NLP Project\Python\pythonProject3\qa_index.ann')

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up BigQuery client
service_account_file = r'D:\NLP Project\Python\pythonProject3\fine-gradient-145319-1d6138310806.json'
credentials = service_account.Credentials.from_service_account_file(service_account_file)
client = bigquery.Client(credentials=credentials)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        search_term = request.form['search']
        query_embedding = model.encode(search_term)
        k = 5
        nearest_indices = index.get_nns_by_vector(query_embedding, k)
        indices_str = ', '.join(map(str, nearest_indices))
        query = f"""
        SELECT answers, question, id
        FROM `fine-gradient-145319.stackoverflow.Similarity`
        WHERE id IN ({indices_str})
        """
        query_job = client.query(query)
        results = []
        for row in query_job:
            results.append({'question': row['question'], 'answers': row['answers']})
        return render_template('index.html', results=results)
    return render_template('index.html')


@app.template_filter('remove_newlines')
def remove_newlines(text):
    if isinstance(text, list):
        return ''.join(re.sub(r'\\n', '', str(item)) for item in text)
    return re.sub(r'\\n', '', text)



if __name__ == '__main__':
    app.run(debug=True)
