from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import psycopg2
import torch
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from scipy.spatial import distance
import logging
import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import numpy as np


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Spacy model
nlp = spacy.load('en_core_web_sm')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpecterModel:
    def __init__(self) -> None:
        self.load_model()

    @classmethod
    def load_model(cls):
        if not hasattr(cls, "initialized"):
            cls.device = torch.device("cpu")

            # initialize model
            cls.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
            cls.model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
            cls.model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
            cls.model = cls.model.to(cls.device)

            cls.initialized = True
    
    def text_to_vector(self, text: str):
        text = [text]

        with torch.no_grad():
            input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512).to(self.device)
            output = self.model(**input)
        
        embedding = output.last_hidden_state[:, 0, :].cpu().detach().numpy()  # Extract the embeddings of the [CLS] token

        # embedding --> 2D np.array of shape (1, 768)
        return embedding


class TfIDFModel:
    def __init__(self) -> None:
        self.load_model()

    @classmethod
    def load_model(cls):
        if not hasattr(cls, "initialized"):
            cls.model = joblib.load('tfidf_vectorizer.joblib')

            cls.initialized = True
    
    def text_to_vector(self, text: str):
        text = [text]

        embedding = self.model.transform(text).toarray()

        # embedding --> 2D np.array of shape (1, 768)
        return embedding


class NearestNei:
    def __init__(self) -> None:
        pass

    @classmethod
    def fit_k_model(cls, k: int, metric: str, vectors: List[List[float]], flag="specter"):
        """
        Valid metrics --> ["manhattan", "l2", "cosine", "sqeuclidean", "canberra", "chebyshev", "correlation"]
        flag --> ["specter", "tf-idf"]
        """
        if not hasattr(cls, "initialiazed_k"):
            cls.initialiazed_k = {}
        
        if f"{k}_{metric}_{flag}" not in cls.initialiazed_k:
            if metric in ["manhattan", "l2", "cosine"]:
                cls.initialiazed_k[f"{k}_{metric}_{flag}"] = NearestNeighbors(
                    n_neighbors=k,
                    metric=metric,
                    n_jobs=-1
                )
            else:
                metric_func = getattr(distance, metric)
                cls.initialiazed_k[f"{k}_{metric}_{flag}"] = NearestNeighbors(
                    n_neighbors=k,
                    metric=metric_func,
                    n_jobs=-1
                )
            
            cls.initialiazed_k[f"{k}_{metric}_{flag}"].fit(vectors)

class SearchResult(BaseModel):
    index: int
    distance: float
    text: str
    keyword_present: bool

class SearchResponse(BaseModel):
    results: List[SearchResult]
    precision: float

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def fetch_data() -> Tuple[List, List, List]:
    conn_params = {
        'dbname': 'advanced_db',
        'user': 'password',
        'password': 'password',
        # 'host': 'localhost',
        'host': 'postgres',
        # 'port': 5433
        'port': 5432
    }

    conn = None
    cur = None

    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()

        # Execute the query to fetch all data from the medical_data table
        cur.execute('SELECT * FROM medical_data')
        
        # Fetch all rows from the executed query
        rows = cur.fetchall()

        # Get column names from the cursor
        ids = [row[0] for row in rows]
        vectors = [row[2] for row in rows]
        vectors_tf_idf = [row[3] for row in rows]
        texts = [row[1] for row in rows]

        return ids, vectors, vectors_tf_idf, texts

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Database error: {error}")
    finally:
        if conn is not None:
            if cur is not None:
                cur.close()
            conn.close()
            print('Database connection closed.')

def preprocess_text(text: str) -> str:
    # Lowercasing
    text = text.lower()

    # Removing Punctuation & Special Characters
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Stop-Words Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    lemmatized_tokens = []
    doc = nlp(' '.join(tokens))
    for token in doc:
        lemmatized_tokens.append(token.lemma_)

    return ' '.join(lemmatized_tokens)

def keyword_in_text(text: str, keywords: str) -> bool:
    preprocessed_text = preprocess_text(text)
    preprocessed_keywords = preprocess_text(keywords)
    keyword_list = preprocessed_keywords.split()
    return any(keyword in preprocessed_text.split() for keyword in keyword_list)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_model=SearchResponse)
async def search(keywords: str = Form(...), k: int = Form(...), metric: str = Form(...)) -> SearchResponse:
    num_data = 4000

    # Fetch data from the database
    ids, vectors, vectors_tf_idf, texts = fetch_data()

    # SPECTER 2
    specter_model = SpecterModel()
    embedding = specter_model.text_to_vector(keywords)

    # TF-IDF
    tf_idf_model = TfIDFModel()
    embedding_tf_idf = tf_idf_model.text_to_vector(keywords)

    nn_model = NearestNei()
    # fit nn model for specter embeddings
    nn_model.fit_k_model(
        # k=k,
        k=num_data,
        metric=metric,
        vectors=vectors,
        flag="specter"
    )
    # fit nn model for tf-idf embeddings
    nn_model.fit_k_model(
        # k=k,
        k=num_data,
        metric=metric,
        vectors=vectors_tf_idf,
        flag="tf-idf"
    )

    # Find the k closest points to the example vector
    distances, indices = nn_model.initialiazed_k[f"{num_data}_{metric}_specter"].kneighbors(embedding)
    distances_tf_idf, indices_tf_idf = nn_model.initialiazed_k[f"{num_data}_{metric}_tf-idf"].kneighbors(embedding_tf_idf)

    # Re-order tf-idf embeddings
    reordering = find_reordering(indices[0], indices_tf_idf[0])
    distances_tf_idf[0] = distances_tf_idf[0][reordering]
    indices_tf_idf[0] = indices_tf_idf[0][reordering]

    # Combine distances
    combined_dist = combined_distances(distances[0], distances_tf_idf[0])

    # Re-sort
    combined_indices = np.argsort(combined_dist)

    # Retrieve the top-k results
    top_k_indices = combined_indices[:k]

    # Prepare the result
    # result = []
    # for idx, i in enumerate(indices[0]):
    #     result.append(SearchResult(
    #         index=ids[i],
    #         distance=distances[0][idx],
    #         text=texts[i],
    #         keyword_present=keyword_in_text(texts[i], keywords)
    #     ))

    result = []
    for idx, i in enumerate(indices[0][top_k_indices]):
        result.append(SearchResult(
            index=ids[i],
            distance=combined_dist[combined_indices[idx]],
            text=texts[i],
            keyword_present=keyword_in_text(texts[i], keywords)
        ))

    # Calculate precision
    precision = len([x for x in result if x.keyword_present]) / len(result)

    return SearchResponse(results=result, precision=precision)


def find_reordering(original, target):
    # Create a dictionary mapping each value in the target to its index
    target_index_map = {value: idx for idx, value in enumerate(target)}
    # Find the positions in the original array that match the target order
    reordering = [target_index_map[value] for value in original]
    return reordering

def min_max_normalize(distances):
    min_val = np.min(distances)
    max_val = np.max(distances)
    return (distances - min_val) / (max_val - min_val)

def combined_distances(distances1, distances2, weight1=0.5, weight2=0.5):
    norm_distances1 = min_max_normalize(distances1)
    norm_distances2 = min_max_normalize(distances2)
    combined = weight1 * norm_distances1 + weight2 * norm_distances2
    return combined


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)   # 127.0.0.2 for testing
