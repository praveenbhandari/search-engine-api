
from fastapi import FastAPI,APIRouter,Body
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
    embeddings=embeddings, sparse_encoder=bm25, index=index,alpha=0.1,top_k=20,
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
    
@my_router.post("/add_query")
async def add_item(item: query_item):
    # print(item.query)
    mycursor.execute("CREATE TABLE IF NOT EXISTS search_queries (id INT AUTO_INCREMENT PRIMARY KEY, query VARCHAR(255), datetime DATETIME)")
    sql = "INSERT INTO search_queries (query, datetime) VALUES (%s, %s)"
    values = (item.query, datetime.now())
    mycursor.execute(sql, values)
    mydb.commit()
    return item

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
        top_k=10,
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
    # return ids,metadatas,scores
    return ids,metadatas,scores


# index.query(vectors=vector,top_k=5)
import string
# response
def process(textt):
  word_tokens = word_tokenize(textt)
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
  filtered_sentence = [w for w in filtered_sentence if w not in string.punctuation]
#   print(filtered_sentence)
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
    # print(bm25.doc_freqs)
    # print(bm25.corpus_size)
    return bm25


def word_frequency_scores(doc, words):
    translator = str.maketrans('', '', string.punctuation)
    normalized_doc = doc.lower().translate(translator)
    doc_words = normalized_doc.split()
    total_words = float(len(doc_words))
    total_score = 0.0
    
    # Calculate term frequency for each word and add to total score
    for word in words:
        normalized_word = word.lower()
        word_count = float(doc_words.count(normalized_word))
        
        if total_words > 0:
            tf = word_count / total_words
        else:
            tf = 0
        total_score += tf
    
    # Calculate the average score if there are words
    if words:
        average_score =  len(words) / total_score
    else:
        average_score = 0

    return (word_count,total_words,average_score)


# Example usage
# paragraph = "This is a sample paragraph. This paragraph is provided as an example. Here, 'paragraph' is mentioned a few times."
# word = "paragraph"
# score = word_frequency_score(paragraph, word)

# print(f"Term Frequency of '{word}' in the paragraph: {score}")


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


def search(docs,query):
    bm25 = tokenize_documents(docs)
    term_weights = predict_term_weights(query)
    weighted_query = [(term, weight) for term, weight in term_weights.items()]
    # print()
    # print(weighted_query)
    scores = defaultdict(float)
    for term, weight in weighted_query:
        term_scores = bm25.get_scores([term])
        # print(term_scores,".......",weight)
        for doc_id, score in enumerate(term_scores):
            # print(score,"--------",weight)
            scores[doc_id] += score * weight
    
    # query_score = [word_frequency_scores(doc, query.split()) for doc in docs]
    query_score = [word_frequency_scores(doc, query.split()) for doc in docs]
    # print(scores)
    # query_score=[]

    

    doc_score_dict = {doc: score for doc, score in zip(docs, query_score)}

    max_score = max(scores.values()) if scores else 0     
    # print(q.)
    # sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # sorted_cases = sorted(scores.items(), key=lambda x: x["Document Date"])
    # sorted_docs_with_scores = [(docs[doc_id], scores) for doc_id, scores in sorted_docs]
    scaled_scores = {doc_id: (score / max_score * 100) if max_score > 0 else 0 for doc_id, score in scores.items()}
    sc=[f"{j:.2f}" for _,j in scaled_scores.items()]
    # for k in scaled_scores.items():
    #     s
    sorted_docs = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
    # qq=sorted(q.items(), key=lambda x: x[1], reverse=True)
    # sorted_cases    
    # print((sorted_docs))
    return sc,[doc_id for doc_id, _ in sorted_docs],[docs[doc_id] for doc_id, _ in sorted_docs],query_score


# def search_and_scale(docs, query):
#     bm25 = tokenize_documents(docs)
#     term_weights = predict_term_weights(query)
#     weighted_query = [(term, weight) for term, weight in term_weights.items()]
#     scores = defaultdict(float)
#     for term, weight in weighted_query:
#         term_scores = bm25.get_scores([term])
#         for doc_id, score in enumerate(term_scores):
#             scores[doc_id] += score * weight

#     # Find the maximum score for scaling
#     max_score = max(scores.values()) if scores else 0

#     # Scale scores to be in the range of 0 to 100
#     scaled_scores = {doc_id: (score / max_score * 100) if max_score > 0 else 0 for doc_id, score in scores.items()}
    
#     # Sort documents by scaled scores in descending order
#     sorted_docs = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)

#     return sorted_docs

def searchh(texttt):
#   print(texttt)
  pine = retriever.get_relevant_documents(str(texttt))
  # print(pine)
  # for i in pine:
  #   print(i.page_content)
  content=[i.page_content for i in pine]
  meta=[i.metadata for i in pine]
#   print(content)
#   print(meta)
  procesedd_text=[process(content[i]+str(meta[i])) for i in range(len(content))]
  return pine,procesedd_text
  
from datetime import datetime

# def results(query):
#     pine,textss = searchh(query)
#     # textss
#     # print(textss)

#     score,ids,res,q_score=search(textss,query)
#     # result=[for i in ids]
#     # print(res)
#     # count=0
#     resultss=[]
#     for i in ids:
#         # print(round(float(score[count])*100+100,2),pine[i])
#         if float(score[i]) < 0:
#             resultss.append((0,pine[i],(0,0,0)))
#         else:
#             resultss.append((score[i],pine[i],q_score[i]))
#         # count += 1
#     # import time
#     # print(resultss)
#     # # print(textss)
#     # ref=[]
#     # metaa=[]
#     # for i in ids[:5]:
#     #     ref=[pine[i].page_content]
#     #     metaa=[pine[i].metadata]
#     #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     #     print(pine[i].page_content)
#     # sorted_cases = sorted(pine.metadata, key=lambda x: x["Document Date"])
   
#     def round_score(score):
#         # print(str(score).split(".")[0])
#     # This is a placeholder for your round_score function
#         # return (int(str(score).split(".")[0]))
#         return round(float(score))

#     def get_date(document):
#         month_dict = {
#             'January': '01',
#             'February': '02',
#             'March': '03',
#             'April': '04',
#             'May': '05',
#             'June': '06',
#             'July': '07',
#             'August': '08',
#             'September': '09',
#             'October': '10',
#             'November': '11',
#             'December': '12'
#         }
#         try:
#             date_str = document.metadata["Date"]
            
#             if date_str == "Not found":
#                 # print(doc)

#                 print("ffffff", document)
#                 # return "00-00-0000"
#                 date_obj = datetime.strptime("12-31-9999", '%m-%d-%Y')
#                 return date_obj


#             print("--------", date_str)
            
#             try:
#                 date_obj = datetime.strptime(date_str, '%m-%d-%Y')
#                 return date_obj
#             except ValueError:
#                 pass
            
#             try:
#                 date_obj = datetime.strptime(date_str, '%m-%B-%Y')
#                 return date_obj
#             except ValueError:
#                 pass
            
#             try:

#                 # date_obj = datetime.strptime(date_str, '%d %B %Y')
#                 parsed_date = datetime.strptime(date_str, "%d %B %Y")

#                 # Get the month name from the parsed date and convert it to its number representation
#                 month_name = parsed_date.strftime("%B")
#                 month_number = month_dict.get(month_name)

#                 # Convert to the desired format: month date year
#                 formatted_date = parsed_date.strftime(f"{month_number}-%d-%Y")
#                 date_obj = datetime.strptime(f"{month_number}-{parsed_date.date}-{parsed_date.year}", '%m-%d-%Y')
#                 print("-----",date_obj)
#                 return date_obj
#             except ValueError:
#                 pass
            
#             if "/" in date_str:
#                 d, m, y = date_str.split("/")
#                 date_str = f"{m}-{d}-{y}"
#             elif "-" in date_str:
#                 d, m, y = date_str.split("-")
#                 date_str = f"{m}-{d}-{y}"
#             else:
#                 print("Date format not recognized", date_str)
#                 return None
            
#             try:
#                 date_obj = datetime.strptime(date_str, '%m-%d-%Y')
#                 return date_obj
#             except ValueError:
#                 print("Date format not recognized", date_str)
#                 parsed_date = datetime.strptime(date_str, "%d %B %Y")

#                 # Get the month name from the parsed date and convert it to its number representation
#                 month_name = parsed_date.strftime("%B")
#                 month_number = month_dict.get(month_name)

#                 # Convert to the desired format: month date year
#                 formatted_date = parsed_date.strftime(f"{month_number}-%d-%Y")
#                 date_obj = datetime.strptime(f"{month_number}-{parsed_date.date}-{parsed_date.year}", '%m-%d-%Y')
#                 print("-----",date_obj)
#                 return date_obj
#                 # return None

#             return None
#         except KeyError:
#             print("Date not found in document metadata")
#             return None


#     grouped_results = {}
#     for score_value, document, q_score_value in resultss:
#         rounded_score = round_score(score_value)
#         if 1 <= rounded_score <= 100:
#             if rounded_score not in grouped_results:
#                 grouped_results[rounded_score] = []
#             grouped_results[rounded_score].append((score_value, document, q_score_value))

#     # Now, sort the documents within each score group by date
#     # for score in grouped_results:
#     #     grouped_results[score] = sorted(grouped_results[score], key=lambda x: get_date(x[1]), reverse=False)

#     for score in grouped_results:
        
#         grouped_results[score] = sorted(grouped_results[score], key=lambda x: get_date(x[1]) if get_date(x[1]) is not None else datetime.max, reverse=False)
#     data=[]
#     for i in grouped_results:
#         # data.append(j)
#         for j in grouped_results.get(i):
#             # print("--------**********")
#             # print(j)
#             data.append(j)
#     return resultss,ids,data

# results("war in iran")
# retriever.get_relevant_documents(str("war"))

def results(query):
    pine, textss = searchh(query)
    score, ids, res, q_score = search(textss, query)
    
    resultss = []
    for i in ids:
        # print("---------",pine[i].metadata["Case Name"])
        if pine[i].metadata["Case Name"] == "Not Available":
                resultss.append(("Relevant:"+score[i], pine[i], q_score[i]))
        else:
            # print("elseeeeee",pine[i].metadata["Case Name"])
        
            if float(score[i]) < 0:
                resultss.append((0, pine[i], (0, 0, 0)))
            else: 
                resultss.append((score[i], pine[i], q_score[i]))
            

    # def round_score(score):
    #     return round(float(score))

    # # def get_date(document):
    # #     month_dict = {
    # #         'January': '01', 'February': '02', 'March': '03', 'April': '04',
    # #         'May': '05', 'June': '06', 'July': '07', 'August': '08',
    # #         'September': '09', 'October': '10', 'November': '11', 'December': '12'
    # #     }
    # #     try:
    # #         date_str = document.metadata.get("Date", "Not found")
    # #         if date_str == "Not found":
    # #             return datetime.strptime("9999-12-31", '%Y-%m-%d')

    # #         for fmt in ['%m-%d-%Y', '%m-%B-%Y', '%d %B %Y']:
    # #             try:
    # #                 if fmt == '%d %B %Y':
    # #                     # For the format '%d %B %Y', parse it directly and convert to desired format
    # #                     parsed_date = datetime.strptime(date_str, fmt)
    # #                     month_number = month_dict.get(parsed_date.strftime("%B"))
    # #                     return datetime.strptime(f"{parsed_date.year}-{month_number}-{parsed_date.day}", '%Y-%m-%d')
    # #                 else:
    # #                     parsed_date = datetime.strptime(date_str, fmt)
    # #                     return parsed_date
    # #             except ValueError:
    # #                 pass

    # #         if "-" in date_str:
    # #             d, m, y = date_str.split("-")
    # #             date_str = f"{m}-{d}-{y}"
    # #             parsed_date = datetime.strptime(date_str, '%m-%d-%Y')
    # #             return parsed_date
    # #         elif "/" in date_str:
    # #             d, m, y = date_str.split("/")
    # #             date_str = f"{m}-{d}-{y}"
    # #             parsed_date = datetime.strptime(date_str, '%m-%d-%Y')
    # #             return parsed_date
    # #         else:
    # #             print("Date format not recognized", date_str)
    # #             return None
    # #     except KeyError:
    # #         print("Date not found in document metadata")
    # #         return None
    # def get_date(document):
    #     month_dict = {
    #         'January': '01', 'February': '02', 'March': '03', 'April': '04',
    #         'May': '05', 'June': '06', 'July': '07', 'August': '08',
    #         'September': '09', 'October': '10', 'November': '11', 'December': '12'
    #     }
    #     try:
    #         date_str = document.metadata.get("Date", "Not found")
    #         if date_str == "Not found":
    #             return "9999-12-31"

    #         for fmt in ['%m-%d-%Y', '%m-%B-%Y', '%d %B %Y']:
    #             try:
    #                 if fmt == '%d %B %Y':
    #                     # For the format '%d %B %Y', parse it directly and convert to desired format
    #                     parsed_date = datetime.strptime(date_str, fmt)
    #                     month_number = month_dict.get(parsed_date.strftime("%B"))
    #                     return f"{parsed_date.year}-{month_number}-{parsed_date.day}"
    #                 else:
    #                     parsed_date = datetime.strptime(date_str, fmt)
    #                     return parsed_date.strftime("%Y-%m-%d")
    #             except ValueError:
    #                 pass

    #         if "-" in date_str:
    #             d, m, y = date_str.split("-")
    #             date_str = f"{m}-{d}-{y}"
    #             return date_str
    #         elif "/" in date_str:
    #             d, m, y = date_str.split("/")
    #             date_str = f"{m}-{d}-{y}"
    #             return date_str
    #         else:
    #             print("Date format not recognized", date_str)
    #             return None
    #     except KeyError:
    #         print("Date not found in document metadata")
    #         return None

    # grouped_results = {}
    # relevant_group = "Relevant"

    # for score_value, document, q_score_value in resultss:
    #     document.metadata["Date"] = get_date(document)  # Set the value of the Date field in document.metadata

    #     case_name = document.metadata["Case Name"]

    #     if case_name == "Not Available":
    #         print(case_name)
    #         # Group records with case name "Not Available" under a relevant group
    #         if relevant_group not in grouped_results:
    #             grouped_results[relevant_group] = []
    #         grouped_results[relevant_group].append((score_value, document, q_score_value))
    #         # print(grouped_results)
    #     else:
    #         rounded_score = round_score(score_value)
    #         if 1 <= rounded_score <= 100:
    #             if rounded_score not in grouped_results:
    #                 grouped_results[rounded_score] = []
    #             grouped_results[rounded_score].append((score_value, document, q_score_value))

    # # Now, sort the documents within each score group by date
    # for group_key in grouped_results:
    #     if group_key != relevant_group:
    #         grouped_results[group_key] = sorted(grouped_results[group_key], key=lambda x: get_date(x[1]) if isinstance(get_date(x[1]), datetime) else datetime.max, reverse=False)
    # # print("========",grouped_results)
    data = []
    # for group_key in grouped_results:
    #     print(group_key)
    #     for item in grouped_results[group_key]:
    #         # print
    #         data.append(item)
    #     # print("--------",data)
    # # print(data)
    # # print(grouped_results.get(relevant_group))
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
    # print(sparse_dict_tokens)
    return sparse_dict_tokens

def querr(query):
    words = [w for w, s in sim_test(query).items() if len(w) > 2]
    wordss=[]
    a=['procedural', 'applicant', 'case', 'citations', 'communication', 'court', 'date', 'decisions', 'details', 'document', 'history', 'id', 'impact', 'involved', 'issue','judges', 'key', 'legal', 'matter', 'parties', 'points', 'principle', 'procedural', 'references', 'representatives', 'rulings','significance', 'situation', 'subject', 'submission', 'substantive', 'summary', 'tribunal', 'victim']

    for i in words:
        # print(i)
        if i in a:
            if i in query.lower().split():
                wordss.append(i)
            continue
        else:
            wordss.append(i)    
        # print(words)
    # lem=[words]
    # print(lem)
    # words_string = " "
    # for i in words[:5]:
    #     words_string = words_string+" "+i

    # print(words_string)
    resultss,ids,q_score=results(query)  
# results("ICC01/12-01/18")  
    return resultss,ids,wordss,q_score

# def lemmatize(words):
#     b=set()
#     for word in words:
#         b.add(word)
#         lemmatized_word = lemmatizer.lemmatize(word, pos="v") 
#         b.add(lemmatized_word)
#         stemmed_word = stemmer.stem(word)
#         b.add(stemmed_word)
#         # print(f"word: {word} Lemmatized: {lemmatized_word}, Stemmed: {stemmed_word}")
#     return b 





@my_router.post("/search")
async def search_res(query: query_item):
    # print(query.query)
    # mycursor.execute("CREATE TABLE IF NOT EXISTS search_queries (id INT AUTO_INCREMENT PRIMARY KEY, query VARCHAR(255), datetime DATETIME)")
    # sql = "INSERT INTO search_queries (query, datetime) VALUES (%s, %s)"
    # values = (query.query, datetime.now())
    # mycursor.execute(sql, values)
    # mydb.commit()
    # response=retriever.get_relevant_documents(str(query.query))
    result,ids,word,q_score=querr(query.query)
    # print((a))
    # for i in a:
    # print(type(lemmatize(a)))
    # response=results(query.query)

    # data=
    # for s,d,_ in resultss:
    #     scores.append(s)
    # for i in 

    # return list([result]),word,q_score
    return result,word,q_score

@my_router.post("/search1")
async def search_res1(query: query_item):
    res=get_embd(query.query)
    # print(res[1])
    return res

@my_router.post("/suggest")
async def sugest(query: query_item):
    res=suggest_model.suggest(query.query)
    # print(res)
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