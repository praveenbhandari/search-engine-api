"""Example LangChain server exposes a retriever."""
from fastapi import FastAPI,APIRouter
from langchain.embeddings import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
# import pinecone
from pinecone import Pinecone
from langserve import add_routes
# from langchain.vectorstores import Pinecone
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
import mysql.connector
from pydantic import BaseModel
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
# import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

import fitz  
import requests

from collections import defaultdict
import openai

openai.api_key = "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"




mydb = mysql.connector.connect(
  host="search-engine.ckkhapdb5dit.ap-south-1.rds.amazonaws.com",
  user="admin",
  password="Praveen123",
  # database="test"
)

mycursor = mydb.cursor()
try:
    mycursor.execute("use search_engine_dev")
except:
    mycursor.execute("CREATE database search_engine_dev")
    mycursor.execute("use search_engine_dev")



index_name = "dev"
openai_api_key="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"
#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
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# initialize pinecone
# pinecone.init(
#     api_key="9db53de5-e4af-4151-a24d-995577de48cf",  # find at app.pinecone.io a242896b-4f43-484a-9d48-a43fa5a71481
#     environment="gcp-starter",  # next to api key in console
# )
pinecone=Pinecone(api_key="9db53de5-e4af-4151-a24d-995577de48cf",  # find at app.pinecone.io a242896b-4f43-484a-9d48-a43fa5a71481
)

index = pinecone.Index(index_name)
bm25= BM25Encoder().default()

processed_context=[]
stop_words = set(stopwords.words('english'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


model_id = 'naver/splade-cocondenser-ensembledistil'

tokenizer1 = AutoTokenizer.from_pretrained(model_id)
model1 = AutoModelForMaskedLM.from_pretrained(model_id)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25, index=index,alpha=0.1,top_k=100,
)

# async def fetch():
#     return index.fetch(ids=["039e0b40c0c879a46c87a7899f8c81343e82781986f3e01ddfd31c71af01030f"], namespace='')

# Create store from existing index
#vectorstore = Pinecone.from_existing_index(index_name, embeddings, "context")

#retriever = vectorstore.as_retriever()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

my_router = APIRouter()
# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream

# @my_router.get("/get_index", tags=["users"])
# async def fetch():
#     print("gfetrtgggg***************")
#     return index.fetch(ids=["id"], namespace='')["vectors"][id]["metadata"]
    

# my_router.add_api_route("/get_index", endpoint=fetch)

# add_routes(app,fetch, path="/get_data")

@my_router.get("/get_index/{id}")
async def read_items(id:str):
    # print(index.fetch(ids=["039e0b40c0c879a46c87a7899f8c81343e82781986f3e01ddfd31c71af01030f"], namespace='')["vectors"]["039e0b40c0c879a46c87a7899f8c81343e82781986f3e01ddfd31c71af01030f"]["metadata"])
    return index.fetch(ids=[id], namespace='')["vectors"][id]["metadata"] 

class query_item(BaseModel):
    query: str
    
# @my_router.post("/add_query")
# async def add_item(item: query_item):
#     print(item.query)
#     mycursor.execute("CREATE TABLE IF NOT EXISTS search_queries (id INT AUTO_INCREMENT PRIMARY KEY, query VARCHAR(255), datetime DATETIME)")
#     sql = "INSERT INTO search_queries (query, datetime) VALUES (%s, %s)"
#     values = (item.query, datetime.now())
#     mycursor.execute(sql, values)
#     mydb.commit()
#     return item

    # try:
        # sql = "INSERT INTO search_queries (query, datetime) VALUES (%s, %s)"
        # values = (query, datetime.now())
        # mycursor.execute(sql, values)
        # mydb.commit()
    # except:
    #     print("table created")
    #     mycursor.execute("CREATE TABLE search_queries (id INT AUTO_INCREMENT PRIMARY KEY, query VARCHAR(255), datetime DATETIME)")
    #     mydb.commit()
    # mycursor.execute("CREATE TABLE search_queries (query VARCHAR(255), time VARCHAR(255))")
# from pinecone import Pinecone

# pc = Pinecone(api_key="9db53de5-e4af-4151-a24d-995577de48cf")
# index = pc.Index("dev")
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
def get_embd(strr):
    response = openai.Embedding.create(
        input=strr,
        model="text-embedding-3-small"
    )
    response1 = index.query(
        vector=response.data[0].embedding,
        top_k=100,
        # include_values=True,
        include_metadata=True
        )
    # print(jsonable_encoder(response1["matches"]))
    ids=[]
    metadatas=[]
    scores=[]
    # sparse_values=[]
    # values=[]
    for i in response1["matches"]:
        # print(i["id"])
        ids.append(i["id"])
        metadatas.append(i["metadata"])
        scores.append(i["score"])
        # sparse_values.append(i["sparse_values"])
        # values.append(i["values"])
        # print(i["metadata"])
        # print(i["score"])
        # print(i["sparse_values"])
        # print(i["values"])
    for metadata, score in zip(metadatas, scores):
        # print(score)
        # print(float(score)*100)
        # print(100-float(score)*100)
        
        metadata["score"] = float(score)*1000
    # return ids,metadatas,scores
    return ids,metadatas


# index.query(vectors=vector,top_k=5)

# response
def process(textt):
  print(textt["context"])
  word_tokens = word_tokenize(textt["context"])
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
  filtered_sentence = []
  for w in word_tokens:
      if w not in stop_words:

          filtered_sentence.append(w)

  a=" ".join(filtered_sentence)
  return a

def tokenize_documents(documents):
    tokenized_docs = [tokenizer.tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25

def predict_term_weights(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    weights = torch.mean(last_hidden_states, dim=1).squeeze().numpy()  # Simplified example
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    return dict(zip(tokens, weights))

def search(docs,query):
    bm25 = tokenize_documents(docs)
    term_weights = predict_term_weights(query)
    weighted_query = [(term, weight) for term, weight in term_weights.items()]

    scores = defaultdict(float)
    for term, weight in weighted_query:
        term_scores = bm25.get_scores([term])
        for doc_id, score in enumerate(term_scores):
            scores[doc_id] += score * weight
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs],[docs[doc_id] for doc_id, _ in sorted_docs]

# def searchh(texttt):
#   pine = retriever.get_relevant_documents(str(texttt))
# #   print(pine)
#   content=[i.page_content for i in pine]
#   meta=[i.metadata for i in pine]
#   # print(content)
#   procesedd_text=[process(i) for i in content]
#   return pine,procesedd_text
  
def searchh1(texttt):
#   pine = retriever.get_relevant_documents(str(texttt))
  pine=get_embd(texttt)
#   print(pine)
  content=[i for i in pine[1]]
#   meta=[i.metadata for i in pine]
  # print(content)
  procesedd_text=[process(i) for i in content]
  return pine,procesedd_text

def results(query):
    pine,textss = searchh1(query)
    ids,res=search(textss,query)
    final_res=[]
    # print(ids)
    for i in ids:
        # print(pine[i])
        final_res.append(pine[i])
    return final_res,ids

# def results(query):
#     pine,textss = searchh1(query)
#     ids,res=search(textss,query)
#     final_res=[]
#     # print(ids)
#     for i in ids:
#         # print(pine[i])
#         final_res.append(pine[i])
#     return final_res,ids

def sim_test(query):
    tokens = tokenizer1(query, return_tensors='pt')
    output = model1(**tokens)
    vec = torch.max(
        torch.log(
            1 + torch.relu(output.logits)
        ) * tokens.attention_mask.unsqueeze(-1),
    dim=1)[0].squeeze()
    cols = vec.nonzero().squeeze().cpu().tolist()
    weights = vec[cols].cpu().tolist()
    # sparse_dict = dict(zip(cols, weights))
    idx2token = {
        idx: token for token, idx in tokenizer.get_vocab().items()
    }
    sparse_dict_tokens = {
        idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
    }
    sparse_dict_tokens = {
        k: v for k, v in sorted(
            sparse_dict_tokens.items(),
            key=lambda item: item[1],
            reverse=True
        )
    }
    return sparse_dict_tokens
# word_lis=[]
word_l=[]
def word_list(query):
    global words
    words = [w for w, s in sim_test(query).items() if len(w) > 2]
    return words

def querr(query):
    words = word_list(query)
    # word_lis = words
    # print(w)
    result,ids=results(query)  
# results("ICC01/12-01/18")  
    return result,words,ids

def lemmatize(words):
    b=set()
    for word in words:
        b.add(word)
        lemmatized_word = lemmatizer.lemmatize(word, pos="v") 
        b.add(lemmatized_word)
        stemmed_word = stemmer.stem(word)
        b.add(stemmed_word)
        # print(f"word: {word} Lemmatized: {lemmatized_word}, Stemmed: {stemmed_word}")
    return b 

def download_pdf(url, local_filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response was an error
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        return True
    except requests.RequestException as e:
        print(f"Failed to download the file: {e}")
        return False

def highlight_words_in_pdf(pdf_path, output_path, word, color=[1, 0, 0]):
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text_instances = page.search_for(word)
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=color)
                highlight.update()
        doc.save(output_path, incremental=True)
        return True
    except Exception as e:
        print(f"Failed to highlight words in the PDF: {e}")
        return False

def main(pdf_url, local_pdf_path, highlighted_pdf_path, word_to_highlight):
    print(pdf_url,local_pdf_path,highlighted_pdf_path,word_to_highlight)

    if download_pdf(pdf_url, local_pdf_path):
        if highlight_words_in_pdf(local_pdf_path, highlighted_pdf_path, word_to_highlight):
            print(f"Highlighted PDF saved as: {highlighted_pdf_path}")
        else:
            print("Failed to highlight words in the PDF.")
    else:
        print("Failed to download the PDF.")



@my_router.post("/urll/{url}")
def pdff(url:str):
    # Example usage
    print("----",url.url)
    pdf_url = url.url
    local_pdf_path = pdf_url.split("/")[-1:]  # Temp file for downloaded PDF
    highlighted_pdf_path = local_pdf_path  # Final output file
    word_to_highlight = word_l  # Example word to highlight
    local_pdf_path = 'downloaded_document.pdf'  # Temp file for downloaded PDF

    # highlighted_pdf_path = 'downloaded_document.pdf'  # Final output file
    print(pdf_url,local_pdf_path,highlighted_pdf_path,word_to_highlight)

    main(pdf_url, 'downloaded_document.pdf', 'downloaded_document.pdf', word_to_highlight)





@my_router.post("/search")
async def search_res(query: query_item):
    # print(query.query)
    # mycursor.execute("CREATE TABLE IF NOT EXISTS search_queries (id INT AUTO_INCREMENT PRIMARY KEY, query VARCHAR(255), datetime DATETIME)")
    # sql = "INSERT INTO search_queries (query, datetime) VALUES (%s, %s)"
    # values = (query.query, datetime.now())
    # mycursor.execute(sql, values)
    # mydb.commit()
    # response=retriever.get_relevant_documents(str(query.query))
    result,word,ids=querr(query.query)
    # print((a))
    # for i in a:
    # print(type(lemmatize(a)))
    # response=results(query.query)
    return result,word,ids

@my_router.post("/search1")
async def search_res1(query: query_item):
    # res=get_embd(query.query)
    # print(res[0])
    # return res
    result,word,ids=querr(query.query)
    # print((a))
    # for i in a:
    # print(type(lemmatize(a)))
    # response=results(query.query)
    return result,word,ids


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:3001","http://your-react-app-origin.com","*","https://search-engine-dev.vercel.app/", "https://search-engine-ui-git-dev-praveenbhandariis-projects.vercel.app/"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(my_router)

add_routes(app, retriever,path="/chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
 