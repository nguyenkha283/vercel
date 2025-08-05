from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import supabase
from openai import OpenAI
import traceback

# === Middleware bắt lỗi in ra log Vercel ===
app = FastAPI()

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print("=== Exception caught ===")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

# === Model request body ===
class Query(BaseModel):
    query: str

# === Khởi tạo kết nối Supabase ===
supabase_client = supabase.create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# === Khởi tạo OpenAI client ===
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/semantic-search")
async def semantic_search(query: Query):
    # Tạo embedding cho query
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query.query
    )
    embedding = response.data[0].embedding

    # Tìm trong Supabase bằng pgvector
    result = supabase_client.rpc(
        "match_documents",
        {"query_embedding": embedding, "match_threshold": 0.7, "match_count": 5}
    ).execute()

    return {"results": result.data}
