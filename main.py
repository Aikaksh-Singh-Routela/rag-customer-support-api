import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from fastembed import TextEmbedding
import faiss
from groq import Groq
from datetime import datetime

# Override deprecated model
os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"

app = FastAPI(title="RAG Customer Support API")

# Initialize Groq client with the NEW model
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"  # <-- UPDATED MODEL

# Initialize embedding model
embedding_model = TextEmbedding(model="BAAI/bge-small-en-v1.5")

# Load FAISS index (create if doesn't exist)
# ... rest of your code ...

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence_scores: List[float]
    timestamp: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    # Your RAG logic here
    # ...
    
    # Call Groq with the correct model
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,  # <-- Uses the updated model
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": request.question}
        ],
        temperature=0.7,
    )
    
    answer = completion.choices[0].message.content
    
    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=["source1", "source2"],
        confidence_scores=[0.8, 0.7],
        timestamp=str(datetime.now())
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)