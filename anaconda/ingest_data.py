'''
@File    :   ingest_data.py
@Time    :   05/2024
@Author  :   nikifori
@Version :   -
'''
import torch
import pandas as pd
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

def main():
    device = torch.device("cpu")
    
    conn_params = {
        'dbname': 'advanced_db',
        'user': 'password',
        'password': 'password',
        # 'host': 'localhost',
        'host': 'postgres',
        # 'port': 5433
        'port': 5432
    }

    # Load dataset
    # csv_path = Path(r"C:\Users\Konstantinos\Desktop\advanced_db\medical_tc_train.csv")
    csv_path = Path(r"medical_tc_train.csv")
    abstract_df = pd.read_csv(csv_path)["medical_abstract"][:4000]
    
    # emb_path = Path(r"C:\Users\Konstantinos\Desktop\advanced_db\text_embeddings_in_order.pt")
    emb_path = Path(r"text_embeddings_in_order.pt")
    embeddings = torch.load(emb_path, map_location=device)[:4000]

    # emb_tf_idf_path = Path(r"C:\Users\Konstantinos\Desktop\advanced_db\tfidf_embeddings.pt")
    emb_tf_idf_path = Path(r"tfidf_embeddings.pt")
    embeddings_tf_idf = torch.load(emb_tf_idf_path, map_location=device)[:4000]

    # Connect to the PostgreSQL database
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()

        # Create the table if it doesn't exist
        cur.execute('''
            CREATE TABLE IF NOT EXISTS medical_data (
                idx SERIAL PRIMARY KEY,
                abstract TEXT,
                embedding DOUBLE PRECISION[],
                embedding_tf_idf DOUBLE PRECISION[]
            );
        ''')

        # Prepare data for batch insertion
        batch_size = 100
        data = [
            (abstract, embedding.tolist(), embedding_tf_idf.tolist())
            for abstract, embedding, embedding_tf_idf in zip(abstract_df, embeddings, embeddings_tf_idf)
        ]

        # Insert data in batches
        print("Inserting data in batches...")
        insert_sql = 'INSERT INTO medical_data (abstract, embedding, embedding_tf_idf) VALUES (%s, %s, %s)'
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            execute_batch(cur, insert_sql, batch)

        # Commit the changes to the database
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            if cur is not None:
                cur.close()
            conn.close()
            print('Database connection closed.')

if __name__ == '__main__':
    main()
