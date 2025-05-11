# Medical Database Searching
## Instructions to run the app

### Clone the repository

`git clone https://github.com/nikifori/AdvancedDB-DWS-AUTH.git ./my_app`

### cd into the folder

`cd .\my_app\`

### Run docker compose file:

`docker compose up --build`

### Open your browser

Go to [http://localhost:8000/](http://localhost:8000/) in your browser.
The first search will last a little longer than usual to complete due to initializations and file‑downloading procedures.

---

## Project Overview

A lightweight web application that demonstrates **semantic search** over 4 000 medical‑paper abstracts.
The system combines deep‑learning embeddings (**SPECTER2**) with classic **TF‑IDF** vectors and performs a *k*-nearest‑neighbours query inside a **PostgreSQL** database to return the most relevant abstracts to a user‑supplied keyword list.

The project showcases:

* Practical use of vector representations inside a relational DBMS.
* A comparison between neural (SPECTER2) and statistical (TF‑IDF) embeddings.
* A fully reproducible, single‑command deployment workflow with **Docker Compose**.

## Assignment Steps (What you will find in this repo)

| #  | Step                                                                                                                                            |
| -- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
|  1 | **Dataset selection** – 4k abstracts sampled from the [Medical‑Abstracts‑TC‑Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus). |
|  2 | **Embedding generation**<br/>• 768‑dim **SPECTER2** vectors (PyTorch).<br/>• 6 144‑dim **TF‑IDF** vectors (scikit‑learn).                       |
|  3 | **Database schema & ingestion** – table with `id`, `raw_text`, `specter2_emb`, `tfidf_emb`; batch loading for speed.                            |
|  4 | **Backend API** – **FastAPI** endpoints for search, health check and docs.                                                                      |
|  5 | **kNN search logic** – distance calculation, normalisation and late fusion (0.5 × SPECTER2 + 0.5 × TF‑IDF).                                     |
|  6 | **Web UI** – minimal HTML served by FastAPI (index, results table).                                                                             |
|  7 | **Containerisation & Orchestration** – multi‑service Compose file (API, Postgres, pgAdmin).                                                     |

## Architecture at a Glance

![image](https://github.com/user-attachments/assets/8fd04a4c-b10e-4eef-8c76-498252a31702)


* **Backend** loads the embeddings into memory on first request, falls back to the cached `.pt`/`.joblib` files on disk.
* kNN is computed with `sklearn.neighbors.NearestNeighbors` (supports 7 distance metrics: *L1*, *L2*, *cosine*, *sqeuclidean*, *canberra*, *chebyshev*, *correlation*).

## Tech Stack

| Category  | Tool                                                                 |
| --------- | -------------------------------------------------------------------- |
| Framework | **FastAPI**, Jinja2 templates                                        |
| ML / NLP  | **PyTorch**, `sentence-transformers`, **SPECTER2**, **scikit‑learn** |
| Data      | **PostgreSQL 14**, `pgvector` extension                              |
| Infra     | **Docker**, **Docker Compose**, **Anaconda**                         |

## Usage

1. Enter one or more **keywords** separated by commas.
2. Choose *k* (5 – 20) and a **distance metric**.
3. Click **Search** and review the returned abstracts. A green ✓ indicates that at least one keyword was found in the text (used for precision calculation).

## Benchmark Results

Average precision over 20 sample queries: **cosine** and **correlation** distances perform best. Precision drops slightly as *k* grows; Chebyshev distance performs worst across the board.

![image](https://github.com/user-attachments/assets/35ea8b21-f4e8-4f50-a699-c25a46fb138a)

