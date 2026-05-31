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
from loguru import logger
import time
from collections import defaultdict
from contextlib import asynccontextmanager

# ============================================
# MONITORING SETUP
# ============================================

# Configure log rotation and retention
logger.add("logs/api.log", rotation="1 day", retention="7 days", compression="zip")
logger.add("logs/errors.log", rotation="1 day", retention="30 days", level="ERROR")

# Metrics tracking
metrics = {
    "total_requests": 0,
    "total_errors": 0,
    "avg_confidence": [],
    "response_times": [],
    "queries_by_hour": defaultdict(int),
}

# ============================================
# MODEL CONFIGURATION
# ============================================
GROQ_MODEL = "llama-3.3-70b-versatile"

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

class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    avg_confidence: float
    avg_response_time: float

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
        start_time = time.time()
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant. Answer based on the provided context. Be concise and helpful."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        elapsed = time.time() - start_time
        logger.info(f"Groq API call completed in {elapsed:.2f} seconds")
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        return f"Error: {str(e)}"

# ============================================
# FASTAPI APP WITH LIFESPAN
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 RAG Customer Support API starting up...")
    logger.info(f"Knowledge base loaded with {len(KNOWLEDGE_BASE)} documents")
    logger.info(f"FAISS index created with dimension: {dimension}")
    yield
    # Shutdown
    logger.info("👋 RAG Customer Support API shutting down...")
    logger.info(f"Final metrics: {metrics['total_requests']} requests processed")

app = FastAPI(title="RAG Customer Support API", version="2.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/")
async def root():
    logger.info("Health check endpoint called")
    return {"message": "RAG Customer Support API", "status": "running", "model": GROQ_MODEL}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question and get an AI-powered answer"""
    
    start_time = time.time()
    request_id = str(int(time.time()))
    
    logger.info(f"[{request_id}] Request received: {request.question[:50]}...")
    
    try:
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
        avg_confidence = np.mean(confidence_scores)
        
        # Build context from sources
        context = "\n".join(sources)
        
        # Generate answer using Groq
        answer = get_groq_answer(request.question, context)
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["avg_confidence"].append(avg_confidence)
        metrics["response_times"].append(time.time() - start_time)
        hour = datetime.now().strftime("%Y-%m-%d %H")
        metrics["queries_by_hour"][hour] += 1
        
        # Log success
        logger.info(f"[{request_id}] Success - Confidence: {avg_confidence:.2%} - Time: {time.time() - start_time:.2f}s")
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            confidence_scores=confidence_scores,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        metrics["total_errors"] += 1
        logger.error(f"[{request_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model": GROQ_MODEL,
        "kb_size": len(KNOWLEDGE_BASE),
        "requests_processed": metrics["total_requests"],
        "errors": metrics["total_errors"],
    }

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API metrics and monitoring data"""
    avg_conf = np.mean(metrics["avg_confidence"]) if metrics["avg_confidence"] else 0
    avg_time = np.mean(metrics["response_times"]) if metrics["response_times"] else 0
    
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        total_errors=metrics["total_errors"],
        avg_confidence=float(avg_conf),
        avg_response_time=float(avg_time),
    )

@app.get("/metrics/detailed")
async def get_detailed_metrics():
    """Get detailed metrics for monitoring dashboard"""
    avg_conf = np.mean(metrics["avg_confidence"]) if metrics["avg_confidence"] else 0
    avg_time = np.mean(metrics["response_times"]) if metrics["response_times"] else 0
    
    return {
        "total_requests": metrics["total_requests"],
        "total_errors": metrics["total_errors"],
        "error_rate": metrics["total_errors"] / max(metrics["total_requests"], 1),
        "avg_confidence": avg_conf,
        "avg_response_time": avg_time,
        "queries_by_hour": dict(metrics["queries_by_hour"]),
        "uptime": "running",
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)