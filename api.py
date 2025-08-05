import os
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from supabase import create_client, Client
from pydantic import BaseModel

# Init Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase_client: Client = create_client(supabase_url, supabase_key)

# Init OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI()

# Allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception middleware
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        traceback.print_exc()
        return {"error": "Internal Server Error"}


# Input schema
class QueryRequest(BaseModel):
    query: str


@app.post("/semantic_search")
async def semantic_search(req: QueryRequest):
    query = req.query

    # Create embedding
    embedding_response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = embedding_response.data[0].embedding[:1536]  # Ensure 1536 dimensions

    # Query Supabase function
    response = supabase_client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 5,
            "match_threshold": 0.78
        }
    ).execute()

    return response.data
