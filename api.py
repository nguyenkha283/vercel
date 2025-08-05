from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
import psycopg2
import os

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/semantic-search")
async def semantic_search(request: QueryRequest):
    query = request.query

    # Lấy embedding từ OpenAI
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    embedding = response["data"][0]["embedding"]
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    # Kết nối Supabase
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor()
    sql = f"""
    SELECT content, source_file, main_heading, sub_heading, embedding <#> '{embedding_str}' AS distance
    FROM documents_segments
    ORDER BY embedding <#> '{embedding_str}' ASC
    LIMIT 5;
    """
    cur.execute(sql)
    results = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "content": r[0],
            "source_file": r[1],
            "main_heading": r[2],
            "sub_heading": r[3],
            "distance": r[4]
        } for r in results
    ]
