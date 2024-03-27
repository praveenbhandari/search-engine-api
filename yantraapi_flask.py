from flask import Flask, request
from flask_cors import CORS,cross_origin
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
import json
# Initialize your Flask app
app = Flask(__name__)

# Configure CORS for your Flask app
CORS(app, resources={r"/chat/*": {"origins": ["http://localhost:3000", "http://localhost:3001", "http://your-react-app-origin.com", "*", "https://search-engine-dev.vercel.app/"]}})

index_name = "dev"
openai_api_key = "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"
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize Pinecone
pinecone.init(
    api_key="9db53de5-e4af-4151-a24d-995577de48cf",
    environment="gcp-starter",
)
index = pinecone.Index(index_name)
bm25 = BM25Encoder().default()

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25, index=index, alpha=0.5, top_k=10
)

# Define a route in Flask. Adjust the routing and request handling as necessary.
@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def handle_chat():
    a=retriever.get_relevant_documents("war")
    print(a)
    return a




@app.route('/get_index/{id}', methods=['GET', 'POST'])
def fetch():

    
    return index.fetch(ids=[id], namespace='')["vectors"][id]["metadata"]
    



# Start the Flask application
if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
