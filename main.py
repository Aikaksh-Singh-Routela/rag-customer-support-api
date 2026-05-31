import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from fastembed import TextEmbedding
import faiss
from groq import Groq
from datetime import datetime
import pickle

app = FastAPI(title="RAG Customer Support API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODEL CONFIGURATION (UPDATED)
# ============================================
GROQ_MODEL = "llama-3.3-70b-versatile"  # Updated from deprecated llama3-70b-8192

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence_scores: List[float]
    timestamp: str

# ============================================
# KNOWLEDGE BASE
# ============================================
KNOWLEDGE_BASE = [
    "Our return policy allows returns within 30 days of purchase with original receipt.",
    "To return an item, visit any store location or use our prepaid shipping label.",
    "Need to cancel an order? Contact us within 2 hours of placing the order.",
    "Password must be at least 12 characters with uppercase, lowercase, numbers, and special characters.",
    "Accounts lock for 15 minutes after 5 failed login attempts.",
    "API tokens expire after 30 minutes. Get a new token by logging in again.",
]

# ============================================
# INITIALIZE EMBEDDINGS AND FAISS
# ============================================
embedding_model = TextEmbedding(model="BAAI/bge-small-en-v1.5")

# Create embeddings for knowledge base
embeddings = list(embedding_model.embed(KNOWLEDGE_BASE))
embeddings_array = np.array([e for e in embeddings], dtype=np.float32)

# Build FAISS index
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# ============================================
# GROQ CLIENT
# ============================================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_groq_answer(question: str, context: str) -> str:
    """Generate answer using Groq Llama 3.3"""
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant. Answer based on the provided context. Be concise and helpful."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/")
async def root():
    return {"message": "RAG Customer Support API", "status": "running", "model": GROQ_MODEL}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question and get an AI-powered answer"""
    
    # Create query embedding
    query_embedding = list(embedding_model.embed([request.question]))[0]
    query_embedding_array = np.array([query_embedding], dtype=np.float32)
    
    # Search FAISS
    distances, indices = index.search(query_embedding_array, k=3)
    
    # Get relevant sources
    sources = [KNOWLEDGE_BASE[idx] for idx in indices[0]]
    
    # Calculate confidence (1 - normalized distance)
    max_distance = np.max(distances[0]) if len(distances[0]) > 0 else 1
    confidence_scores = [1 - (d / (max_distance + 1e-6)) for d in distances[0]]
    
    # Build context from sources
    context = "\n".join(sources)
    
    # Generate answer using Groq
    answer = get_groq_answer(request.question, context)
    
    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        confidence_scores=confidence_scores,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": GROQ_MODEL, "kb_size": len(KNOWLEDGE_BASE)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
