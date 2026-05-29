"""
RAG Customer Support API - FastAPI Implementation
Uses fastembed for lightweight, memory-efficient embeddings
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastembed import TextEmbedding
import faiss
import numpy as np
from groq import Groq
import os
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY environment variable not set!")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Initialize lightweight embedding model (no PyTorch!)
print("🔄 Loading fastembed model (lightweight)...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("✅ Embedding model loaded")

# Knowledge base documents
documents = [
    "Our return policy allows returns within 30 days of purchase with original receipt.",
    "To return an item, visit any store location or use our prepaid shipping label.",
    "Shipping is free on orders over $50. Standard shipping takes 3-5 business days.",
    "Customer support is available 24/7 via live chat and email at support@example.com.",
    "We offer a 1-year warranty on all electronics. Warranty covers manufacturing defects.",
    "You can track your order using the tracking number sent to your email.",
    "Gift cards never expire and can be used online or in stores.",
    "Price match guarantee: We'll match any competitor's price within 14 days of purchase.",
    "Need to cancel an order? Contact us within 2 hours of placing the order.",
    "Join our loyalty program to earn points on every purchase - 100 points = $1 off."
]

# Create embeddings and FAISS index
print("🔄 Creating embeddings and search index...")
document_embeddings = np.array(list(embedding_model.embed(documents)))
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings.astype('float32'))
print(f"✅ FAISS index created with {index.ntotal} documents")

# Create FastAPI app
app = FastAPI(
    title="RAG Customer Support API",
    description="AI-powered customer support that answers questions based on your knowledge base",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# PYDANTIC MODELS
# ============================================
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence_scores: List[float]
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    index_size: int
    model_loaded: bool

# ============================================
# HELPER FUNCTIONS
# ============================================
def search(query: str, k: int = 3):
    query_embedding = np.array(list(embedding_model.query_embed(query))[0])
    # FAISS expects a 2D array: reshape from (384,) to (1, 384)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        similarity = 1 / (1 + dist)
        results.append({
            "content": documents[idx],
            "similarity": similarity,
            "index": idx
        })
    return results

def generate_answer(question: str, context_docs: List[dict]) -> str:
    """Generate a natural answer using Groq LLM"""
    if not client:
        return "API key not configured. Please set GROQ_API_KEY environment variable."
    
    context = "\n\n".join([f"Source {i+1}: {doc['content']}" for i, doc in enumerate(context_docs)])
    
    prompt = f"""You are a friendly, helpful customer support assistant.

Use ONLY this information to answer:

{context}

Customer: {question}

Answer concisely (2-3 sentences). Be helpful and friendly."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/", response_model=dict)
def root():
    return {
        "message": "RAG Customer Support API",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    relevant_docs = search(request.question, k=3)
    
    if not relevant_docs or relevant_docs[0]["similarity"] < 0.3:
        return AnswerResponse(
            question=request.question,
            answer="I couldn't find relevant information in our knowledge base.",
            sources=[],
            confidence_scores=[],
            timestamp=datetime.now().isoformat()
        )
    
    answer = generate_answer(request.question, relevant_docs)
    
    return AnswerResponse(
        question=request.question,
        answer=answer,
        sources=[doc["content"] for doc in relevant_docs],
        confidence_scores=[doc["similarity"] for doc in relevant_docs],
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        documents_loaded=len(documents),
        index_size=index.ntotal,
        model_loaded=True
    )

@app.get("/documents", response_model=List[str])
def list_documents():
    return documents

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 Starting RAG Customer Support API on port {port}")
    print(f"📚 Loaded {len(documents)} documents")
    print(f"📖 API Documentation: http://localhost:{port}/docs")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=port)