
from fastapi import FastAPI,APIRouter,Body
from langchain.embeddings import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from langserve import add_routes
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
from nltk.stem import WordNetLemmatizer, PorterStemmer

from collections import defaultdict
import openai

import pickle
 
openai.api_key = "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"


with open('final.pickle', 'rb') as handle:
    suggest_model = pickle.load(handle)




mydb = mysql.connector.connect(
  host="search-engine.ckkhapdb5dit.ap-south-1.rds.amazonaws.com",
  user="admin",
  password="Praveen123",
)

mycursor = mydb.cursor()
try:
    mycursor.execute("use search_engine_dev")
except:
    mycursor.execute("CREATE database search_engine_dev")
    mycursor.execute("use search_engine_dev")



index_name = "searchengine"
openai_api_key="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"
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

pinecone=Pinecone(api_key="90c1d9cc-c37f-43d2-96f9-c9e7b9e53e92",)
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

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

my_router = APIRouter()
@my_router.get("/get_index/{id}")
async def read_items(id:str):
    return index.fetch(ids=[id], namespace='')["vectors"][id]["metadata"] 

class query_item(BaseModel):
    query: str
    
@my_router.post("/add_query")
async def add_item(item: query_item):
    mycursor.execute("CREATE TABLE IF NOT EXISTS search_queries (id INT AUTO_INCREMENT PRIMARY KEY, query VARCHAR(255), datetime DATETIME)")
    sql = "INSERT INTO search_queries (query, datetime) VALUES (%s, %s)"
    values = (item.query, datetime.now())
    mycursor.execute(sql, values)
    mydb.commit()
    return item

def get_embd(strr):
    response = openai.Embedding.create(
        input=strr,
        model="text-embedding-3-small"
    )
    response1 = index.query(
        vector=response.data[0].embedding,
        top_k=10,
        include_metadata=True
        )
    ids=[]
    metadatas=[]
    scores=[]
    for i in response1["matches"]:
        ids.append(i["id"])
        metadatas.append(i["metadata"])
        scores.append(i["score"])
    return ids,metadatas,scores


import string
def process(textt):
  word_tokens = word_tokenize(textt)
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
  filtered_sentence = [w for w in filtered_sentence if w not in string.punctuation]
  for w in word_tokens:
      if w not in stop_words:
          filtered_sentence.append(w)
  a=" ".join(filtered_sentence)
  return a

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import torch
import torch.nn.functional as F
model = BertModel.from_pretrained('bert-base-uncased')
def tokenize_documents(documents):
    tokenized_docs = [tokenizer.tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25


def word_frequency_scores(doc, words):
    translator = str.maketrans('', '', string.punctuation)
    normalized_doc = doc.lower().translate(translator)
    doc_words = normalized_doc.split()
    total_words = float(len(doc_words))
    total_score = 0.0
    for word in words:
        normalized_word = word.lower()
        word_count = float(doc_words.count(normalized_word))
        
        if total_words > 0:
            tf = word_count / total_words
        else:
            tf = 0
        total_score += tf
    average_score=0
    if total_score > 0.0:
        if words:
            average_score =  len(words) / total_score
        else:
            average_score = 0

    return (word_count,total_words,average_score)



def predict_term_weights(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    weights = torch.mean(last_hidden_states, dim=1).squeeze()
    weights_softmax = F.softmax(weights, dim=0)
    weights_softmax_np = weights_softmax.numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    return dict(zip(tokens, weights_softmax_np))

def search(docs, query):
    bm25 = tokenize_documents(docs)

    term_weights = predict_term_weights(query)

    scores = defaultdict(float)
    for term, weight in term_weights.items():
        term_scores = bm25.get_scores([term])
        for doc_id, score in enumerate(term_scores):
            scores[doc_id] += score * weight

    query_score = [word_frequency_scores(doc, query.split()) for doc in docs]

    max_score = max(scores.values()) if scores else 0
    scaled_scores = {doc_id: (score / max_score * 100) if max_score > 0 else 0 for doc_id, score in scores.items()}

    sorted_docs = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)

    return [f"{j:.2f}" for _, j in scaled_scores.items()], [doc_id for doc_id, _ in sorted_docs], [docs[doc_id] for doc_id, _ in sorted_docs], query_score


def searchh(texttt):
  pine = retriever.get_relevant_documents(str(texttt))
  content=[i.page_content for i in pine]
  meta=[i.metadata for i in pine]
  procesedd_text=[process(content[i]+str(meta[i])) for i in range(len(content))]
  return pine,procesedd_text
  
from datetime import datetime

def results(query):
    pine, textss = searchh(query)
    score, ids, res, q_score = search(textss, query)
    def get_date(document):
        month_dict = {
            'January': '01', 'February': '02', 'March': '03', 'April': '04',
            'May': '05', 'June': '06', 'July': '07', 'August': '08',
            'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
        try:
            date_str = document.metadata.get("Date", "Not found")
            if date_str == "Not found":
                return "0000-01-01"

            for fmt in ['%m-%d-%Y', '%m-%B-%Y', '%d %B %Y']:
                try:
                    if fmt == '%d %B %Y':
                        parsed_date = datetime.strptime(date_str, fmt)
                        month_number = month_dict.get(parsed_date.strftime("%B"))
                        return f"{parsed_date.year}-{month_number}-{parsed_date.day}"
                    else:
                        parsed_date = datetime.strptime(date_str, fmt)
                        return parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    pass

            if "-" in date_str:
                d, m, y = date_str.split("-")
                date_str = f"{m}-{d}-{y}"
                return date_str
            elif "/" in date_str:
                d, m, y = date_str.split("/")
                date_str = f"{m}-{d}-{y}"
                return date_str
            else:
                print("Date format not recognized", date_str)
                return "11-11-11"
        except KeyError:
            print("Date not found in document metadata")
            return None


    resultss = []
    for i in ids:
        if pine[i].metadata["Case Name"] == "Not Available" or pine[i].metadata["Case Name"] == "Not found"  :
                resultss.append(("Relevant:"+score[i], pine[i], q_score[i]))
        else:
            if float(score[i]) < 0:
                resultss.append((0, pine[i], (0, 0, 0)))
            else: 
                resultss.append((score[i], pine[i], q_score[i]))
    
    for score_value, document, q_score_value in resultss:
        document.metadata["Date"] = get_date(document)  
    data = []
    return resultss, ids, data

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

def querr(query):
    words = [w for w, s in sim_test(query).items() if len(w) > 2]
    
    wordss=[]
    a=['procedural', 'applicant', 'case', 'citations', 'communication', 'court', 'date', 'decisions', 'details', 'document', 'history', 'id', 'impact', 'involved', 'issue','judges', 'key', 'legal', 'matter', 'parties', 'points', 'principle', 'procedural', 'references', 'representatives', 'rulings','significance', 'situation', 'subject', 'submission', 'substantive', 'summary', 'tribunal', 'victim']

    for i in words:
        if i in a:
            if i in query.lower().split():
                wordss.append(i)
            continue
        else:
            wordss.append(i)    
    wordss.append(query)
    print(wordss)
    resultss,ids,q_score=results(query)  
    return resultss,ids,wordss,q_score

@my_router.post("/search")
async def search_res(query: query_item):
    result,ids,word,q_score=querr(query.query)
    return result,word,q_score

@my_router.post("/search1")
async def search_res1(query: query_item):
    res=get_embd(query.query)
    return res

@my_router.post("/suggest")
async def sugest(query: query_item):
    res=suggest_model.suggest(query.query)
    return res


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
    uvicorn.run(app, host="localhost", port=8000)
