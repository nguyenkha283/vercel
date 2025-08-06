import os
import traceback
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from supabase import create_client, Client
from pydantic import BaseModel

# Init Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client: Client = create_client(supabase_url, supabase_key)

# Init OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI()

# Allow all origins (for testing/dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception middleware for debug
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        traceback.print_exc()
        return {"error": "Internal Server Error"}

# Healthcheck route
@app.get("/")
def root():
    return {"status": "API is running"}


# Auth helper
def check_auth(authorization: str):
    expected_token = os.getenv("AUTH_TOKEN")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token")


# Input schema
class QueryRequest(BaseModel):
    query: str


# Main route
@app.post("/semantic-search")
async def semantic_search(
    req: QueryRequest,
    authorization: str = Header(None)
):
    # Check token
    check_auth(authorization)

    # Create embedding
    query = req.query
    embedding_response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = embedding_response.data[0].embedding[:1536]  # Ensure 1536 dimensions

    # Query Supabase match_documents function
    response = supabase_client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 5,
            "match_threshold": 0.78
        }
    ).execute()

    return response.data

