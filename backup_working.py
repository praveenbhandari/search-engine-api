
from fastapi import FastAPI,APIRouter,Body
from langchain_openai import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from langserve import add_routes
from langchain_community.retrievers import PineconeHybridSearchRetriever
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
import threading
import queue
from openai import OpenAI
from nltk.corpus import wordnet

from collections import defaultdict
import openai

import pickle
 
openai.api_key = "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"


with open('final11.pickle', 'rb') as handle:
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



index_name = "dev"
openai_api_key="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"
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

pinecone=Pinecone(api_key="9db53de5-e4af-4151-a24d-995577de48cf",  )

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
    embeddings=embeddings, sparse_encoder=bm25, index=index,alpha=0.1,top_k=20,
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

# def get_embd(strr):
#     response = openai.Embedding.create(
#         input=strr,
#         model="text-embedding-3-small"
#     )
#     response1 = index.query(
#         vector=response.data[0].embedding,
#         top_k=10,
#         include_metadata=True
#         )
#     ids=[]
#     metadatas=[]
#     scores=[]
#     for i in response1["matches"]:
#         ids.append(i["id"])
#         metadatas.append(i["metadata"])
#         scores.append(i["score"])
#     return ids,metadatas,scores


def api_call_worker(api_queue, _userPrompt, client, _systemPrompt):
    try:
        print("API call worker started.")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _systemPrompt},
                {"role": "user", "content": _userPrompt}
            ]
        )
        print("API call successful, putting response in queue.")
        api_queue.put(response.choices[0].message.content)
    except Exception as e:
        print(f"Exception in API call worker: {e}")
        api_queue.put(e)

def generate_synonyms(word, max_retries=3, timeout=60):
    OPENAI_API_KEY = "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"  # Replace with your actual API key
    client = OpenAI(api_key=OPENAI_API_KEY)

    _systemPrompt = """
    You are Google, thesaurus, your job is to give me one word for the given word or give the synonym.
      
      JSON SCHEMA
     {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            
            "words": {
                "type": "array",
                    "items": {
                        "type": "string"
                    },
                "minItems": 20,
                "maxItems": 30
            },
        }
    }
      """

    _userPrompt = f"Give me synonyms of{word} in JSON Output"

    retries = 0
    while retries < max_retries:
        # print(f"Main thread: Starting attempt {retries + 1} of {max_retries}")
        api_queue = queue.Queue()
        api_thread = threading.Thread(target=api_call_worker, args=(api_queue, _userPrompt, client, _systemPrompt))
        api_thread.start()

        try:
            # Wait for response with timeout
            result = api_queue.get(timeout=timeout)
            if isinstance(result, Exception):
                raise result
            print("Main thread: Received successful response from API thread.")
            return result
        except queue.Empty:
            print(f"Main thread: Timeout reached after {timeout} seconds. Retrying...")
        except Exception as e:
            print(f"Main thread: Exception occurred: {e}")
        finally:
            retries += 1
            api_thread.join(timeout=10)  # Wait max 10 seconds for the thread to finish
            if api_thread.is_alive():
                print("Main thread: API call thread did not finish within the timeout. Continuing to next attempt.")
            else:
                print("Main thread: API call thread joined.")

    print("Main thread: Maximum retries reached. Exiting.")
    return None





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

# import string

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
    
    if total_score > 0:  # Add a check to avoid division by zero
        average_score = len(words) / total_score
    else:
        average_score = 0

    return (word_count, total_words, average_score)

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
    weighted_query = [(term, weight) for term, weight in term_weights.items()]
    scores = defaultdict(float)
    
    # Check for exact match with quotes
    if '"' in query:
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        for phrase in quoted_phrases:
            for term, weight in weighted_query:
                if term.lower() in phrase.lower():
                    term_scores = bm25.get_scores([term])
                    for doc_id, score in enumerate(term_scores):
                        scores[doc_id] += score * weight * 2  # Double the score for words in quotes
    
    # If no exact match found, prioritize values inside quotes
    if not scores and '"' in query:
        for term, weight in weighted_query:
            if term.startswith('"') and term.endswith('"'):
                term_scores = bm25.get_scores([term])
                for doc_id, score in enumerate(term_scores):
                    scores[doc_id] += score * weight * 3  # Triple the score for values inside quotes

    # If no match found, assign default priority
    if not scores:
        # Split the query into individual terms and check their presence in documents
        for term, weight in weighted_query:
            term_scores = bm25.get_scores([term])
            for doc_id, score in enumerate(term_scores):
                scores[doc_id] += score * weight  # Default priority
    
    # Calculate scores based on word frequency
    query_score = [word_frequency_scores(doc, query.split()) for doc in docs]

    max_score = max(scores.values()) if scores else 0
    scaled_scores = {doc_id: (score / max_score * 100) if max_score > 0 else 0 for doc_id, score in scores.items()}
    sc = [f"{j:.2f}" for _, j in scaled_scores.items()]
    sorted_docs = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
    print(sorted_docs)
    return sc, [doc_id for doc_id, _ in sorted_docs], [docs[doc_id] for doc_id, _ in sorted_docs], query_score



# def search(docs, query):
#     bm25 = tokenize_documents(docs)
#     term_weights = predict_term_weights(query)
#     weighted_query = [(term, weight) for term, weight in term_weights.items()]
#     scores = defaultdict(float)
    
#     # Check if the exact given query matches any document
#     exact_match_doc_ids = [i for i, doc in enumerate(docs) if query.lower() in doc.lower()]
#     if exact_match_doc_ids:
#         for doc_id in exact_match_doc_ids:
#             scores[doc_id] += 100  # Assign highest priority for exact match

#     exact_match_doc_ids = [i for i, doc in enumerate(docs) if query.replace('"',"").lower() in doc.lower()]
#     if exact_match_doc_ids:
#         for doc_id in exact_match_doc_ids:
#             scores[doc_id] += 100  # Assign highest priority for exact match
    
#     # If no exact match, prioritize words in quotes
#     if not exact_match_doc_ids:
#         quoted_phrases = re.findall(r'"([^"]*)"', query)
#         for phrase in quoted_phrases:
#             for term, weight in weighted_query:
#                 if term.lower() in phrase.lower():
#                     term_scores = bm25.get_scores([term])
#                     for doc_id, score in enumerate(term_scores):
#                         scores[doc_id] += score * weight * 2  # Double the score for words in quotes
    
#     # If no match found, prioritize values inside quotes
#     if not scores:
#         for term, weight in weighted_query:
#             if term.startswith('"') and term.endswith('"'):
#                 term_scores = bm25.get_scores([term])
#                 for doc_id, score in enumerate(term_scores):
#                     scores[doc_id] += score * weight * 3  # Triple the score for values inside quotes
#             else:
#                 term_scores = bm25.get_scores([term])
                
#                 for doc_id, score in enumerate(term_scores):
#                     scores[doc_id] += score * weight


#     # Calculate scores based on word frequency
#     query_score = [word_frequency_scores(doc, query.split()) for doc in docs]

#     max_score = max(scores.values()) if scores else 0
#     scaled_scores = {doc_id: (score / max_score * 100) if max_score > 0 else 0 for doc_id, score in scores.items()}
#     sc = [f"{j:.2f}" for _, j in scaled_scores.items()]
#     sorted_docs = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
#     return sc, [doc_id for doc_id, _ in sorted_docs], [docs[doc_id] for doc_id, _ in sorted_docs], query_score

# def search(docs, query):
#     bm25 = tokenize_documents(docs)
#     term_weights = predict_term_weights(query)
#     weighted_query = [(term, weight) for term, weight in term_weights.items()]
#     scores = defaultdict(float)
    
#     # Check for exact match without quotes
#     exact_match_doc_ids = [i for i, doc in enumerate(docs) if query.lower() in doc.lower()]
#     if exact_match_doc_ids:
#         for doc_id in exact_match_doc_ids:
#             scores[doc_id] += 100  # Assign highest priority for exact match
    
#     # Check for exact match with quotes
#     if '"' in query:
#         quoted_phrases = re.findall(r'"([^"]*)"', query)
#         for phrase in quoted_phrases:
#             for term, weight in weighted_query:
#                 if term.lower() in phrase.lower():
#                     term_scores = bm25.get_scores([term])
#                     for doc_id, score in enumerate(term_scores):
#                         scores[doc_id] += score * weight * 2  # Double the score for words in quotes
    
#     # If no exact match found, prioritize values inside quotes
#     if not scores and '"' in query:
#         for term, weight in weighted_query:
#             if term.startswith('"') and term.endswith('"'):
#                 term_scores = bm25.get_scores([term])
#                 for doc_id, score in enumerate(term_scores):
#                     scores[doc_id] += score * weight * 3  # Triple the score for values inside quotes

#     # If no match found, assign default priority
#     if not scores:
#         for term, weight in weighted_query:
#             term_scores = bm25.get_scores([term])
#             for doc_id, score in enumerate(term_scores):
#                 scores[doc_id] += score * weight  # Default priority
    
#     # Calculate scores based on word frequency
#     query_score = [word_frequency_scores(doc, query.split()) for doc in docs]

#     max_score = max(scores.values()) if scores else 0
#     scaled_scores = {doc_id: (score / max_score * 100) if max_score > 0 else 0 for doc_id, score in scores.items()}
#     sc = [f"{j:.2f}" for _, j in scaled_scores.items()]
#     sorted_docs = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
#     print(sorted_docs)
#     return sc, [doc_id for doc_id, _ in sorted_docs], [docs[doc_id] for doc_id, _ in sorted_docs], query_score

# def search(docs,query):
#     bm25 = tokenize_documents(docs)
#     term_weights = predict_term_weights(query)
#     weighted_query = [(term, weight) for term, weight in term_weights.items()]
#     scores = defaultdict(float)
#     for term, weight in weighted_query:
#         term_scores = bm25.get_scores([term])
#         for doc_id, score in enumerate(term_scores):
#             scores[doc_id] += score * weight
    
#     query_score = [word_frequency_scores(doc, query.split()) for doc in docs]

    

#     # doc_score_dict = {doc: score for doc, score in zip(docs, query_score)}

#     max_score = max(scores.values()) if scores else 0     
#     scaled_scores = {doc_id: (score / max_score * 100) if max_score > 0 else 0 for doc_id, score in scores.items()}
#     sc=[f"{j:.2f}" for _,j in scaled_scores.items()]
#     sorted_docs = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
#     return sc,[doc_id for doc_id, _ in sorted_docs],[docs[doc_id] for doc_id, _ in sorted_docs],query_score

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
                return "9999-12-31"

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
                print(date_str)
                try:
                    d, m, y = date_str.split("-")
                except:
                    date_str="00-00-0000"
                    d, m, y = date_str.split("-")
                date_str = f"{m}-{d}-{y}"
                return date_str
            elif "/" in date_str:
                d, m, y = date_str.split("/")
                date_str = f"{m}-{d}-{y}"
                return date_str
            else:
                print("Date format not recognized", date_str)
                return None
        except KeyError:
            print("Date not found in document metadata")
            return "00-00-0000"


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
     # Prioritize words within double quotes
    # sparse_dict_tokens = {
    #     token: weight for token, weight in sparse_dict_tokens.items() if '#' not in token
    # }
    sparse_dict_tokens = {
        token: weight for token, weight in sparse_dict_tokens.items() if '##' not in token
    }

    if query.startswith('"') and query.endswith('"'):
        quoted_query = query[1:-1]  # Remove the double quotes
        if quoted_query in sparse_dict_tokens:
            sparse_dict_tokens[quoted_query] *= 2  # Double the weight of the word

    # Sort the tokens based on weights in descending order
    # sorted_tokens = sorted(
    #     sparse_dict_tokens.items(),
    #     key=lambda item: item[1],
    #     reverse=True
    # )
    sparse_dict_tokens = {
    k: v for k, v in sorted(
        sparse_dict_tokens.items(),
        key=lambda item: item[1],
        reverse=True
    )
}

    # print(sparse_dict_tokens)
    return sparse_dict_tokens
    # return sorted_tokens
import json

import spacy
# !pip install spacy
# !python -m spacy download en
# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def sim_test11(query):
    # Process the query text with spaCy
    doc = nlp(query)
    # Extract relevant tokens based on POS tags
    relevant_tokens = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]

    # Remove empty strings
    relevant_tokens = [token for token in relevant_tokens if token]

    return relevant_tokens

def querr(query):
    # ww = [w for w, s in sim_test(query).items() if len(w) > 2]
    ww=list(json.loads(generate_synonyms(query))["words"])
    # print("===",words)
    ww.append(query)
    words=[]
    # for i in query.split(" "):
    #     words.append(i.lower())
    wordss=[]
    a=['procedural', 'applicant', 'case', 'citations', 'communication', 'court', 'date', 'decisions', 'details', 'document', 'history', 'id', 'impact', 'involved', 'issue','judges', 'key', 'legal', 'matter', 'parties', 'points', 'principle', 'procedural', 'references', 'representatives', 'rulings','significance', 'situation', 'subject', 'submission', 'substantive', 'summary', 'tribunal', 'victim']

    for i in words:
        if i in a:
            # if i in query.lower().split():
            #     wordss.append(i)
            continue
        else:
            wordss.append(i)    
    for i in ww:
        if i not in stop_words:
            print(i)
            wordss.append(i.lower())
    resultss,ids,q_score=results(query)  
    return resultss,ids,wordss,q_score

# def process_words(words):
#     lis=[]
#     pattern = r'"([^"]*)"|\S+'
#     extracted_words = re.findall(pattern, ' '.join(words))
#     if extracted_words:
#         lis.extend(word.strip('"') for word in extracted_words if word)
#     else:
#         lis.extend(words)
#     return lis

import re
@my_router.post("/search")
async def search_res(query: query_item):
    result,ids,word,q_score=querr(query.query)
    # words = re.findall(r'(\b\w+\b|"\w+\s\w+")', query)

    # process=process_words(word)
    pattern = r'"([^"]*)"|\S+'
    extracted_words = re.findall(pattern, ' '.join(word))
    word.extend(word.strip('"') for word in extracted_words if word)
    word.append(query.query.replace('"', ''))
    print("----",word)

    return result,word,q_score

@my_router.post("/suggest")
async def sugest(query: query_item):
    res=suggest_model.suggest(query.query)
    return res

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:3001","http://your-react-app-origin.com","*","https://search-engine-dev.vercel.app/", "https://search-engine-ui-git-dev-praveenbhandariis-projects.vercel.app/"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(my_router)

add_routes(app, retriever,path="/chat")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
