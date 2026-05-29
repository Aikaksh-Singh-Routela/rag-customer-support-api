"""
RAG Customer Support System with fastembed (lightweight)
- No PyTorch needed - runs on ONNX runtime
"""

from fastembed import TextEmbedding
import faiss
import numpy as np
from groq import Groq
import os

# ============================================
# CONFIGURATION
# ============================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("⚠️ Please set your GROQ_API_KEY environment variable!")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)

# Sample documents
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

# Initialize lightweight embedding model
print("🔄 Loading fastembed model...")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
document_embeddings = np.array(list(embedding_model.embed(documents)))
print(f"✅ Loaded {len(documents)} documents\n")

# Build FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings.astype('float32'))
print(f"✅ Built search index with {index.ntotal} vectors\n")

def search(query, k=3):
    query_embedding = np.array(list(embedding_model.query_embed(query))[0])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "content": documents[idx],
            "similarity": 1 / (1 + dist)
        })
    return results

def generate_answer(question, context_docs):
    context = "\n\n".join([f"- {doc['content']}" for doc in context_docs])
    
    prompt = f"""You are a helpful customer support assistant.

Use ONLY this information to answer:

{context}

Customer: {question}

Answer concisely (2-3 sentences)."""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    return completion.choices[0].message.content

def answer_question(question):
    print(f"\n📝 Question: {question}")
    context_docs = search(question)
    
    print(f"📖 Retrieved {len(context_docs)} documents:")
    for doc in context_docs:
        print(f"   {doc['similarity']:.1%} match: {doc['content'][:60]}...")
    
    print("🤖 Generating answer...")
    answer = generate_answer(question, context_docs)
    print(f"\n💡 Answer: {answer}")
    return answer

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 RAG CUSTOMER SUPPORT SYSTEM (fastembed)")
    print("=" * 60)
    
    test_questions = [
        "How do I return an item?",
        "Is shipping free?",
        "What's your warranty on electronics?",
        "How can I track my order?"
    ]
    
    for q in test_questions:
        answer_question(q)
        print("-" * 60)