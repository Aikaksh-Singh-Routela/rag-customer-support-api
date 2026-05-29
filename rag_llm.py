"""
RAG Customer Support System with LLM Integration
- Retrieves relevant documents (FAISS)
- Generates natural answers (Groq LLM)
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os

# ============================================
# IMPORTANT: Replace with your actual Groq API key
# Get it from: https://console.groq.com
# ============================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Check if API key is set
if GROQ_API_KEY == "YOUR_API_KEY_HERE":
    print("⚠️ WARNING: Please set your Groq API key!")
    print("1. Go to https://console.groq.com")
    print("2. Create an API key")
    print("3. Copy and paste it above")
    exit(1)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Sample documents (knowledge base)
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

# Initialize embedding model
print("🔄 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = model.encode(documents)
print(f"✅ Loaded {len(documents)} documents\n")

# Build FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings.astype('float32'))
print(f"✅ Built search index with {index.ntotal} vectors\n")

def search(query, k=3):
    """Find relevant documents for a query"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "content": documents[idx],
            "similarity": 1 / (1 + dist)
        })
    return results

def generate_answer(question, context_docs):
    """Generate a natural answer using Groq LLM"""

    # Prepare context from retrieved documents
    context = "\n\n".join([f"- {doc['content']}" for doc in context_docs])

    # Create prompt
    prompt = f"""You are a helpful customer support assistant for an e-commerce company.

Use ONLY the following information to answer the customer's question.
If the information doesn't contain the answer, say "I don't have that information in our knowledge base."

RELEVANT INFORMATION:
{context}

CUSTOMER QUESTION: {question}

INSTRUCTIONS:
1. Answer naturally and conversationally
2. Be helpful and friendly
3. Cite specific policies when relevant
4. Keep answers concise (2-3 sentences)

YOUR ANSWER:"""

    # Call Groq API
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Free, fast, capable
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )

    return completion.choices[0].message.content

def answer_question(question):
    """Complete RAG pipeline: Retrieve → Generate"""
    print(f"\n📝 Question: {question}")

    # Step 1: Retrieve relevant documents
    context_docs = search(question)

    print(f"📖 Retrieved {len(context_docs)} documents:")
    for doc in context_docs:
        print(f"   {doc['similarity']:.1%} match: {doc['content'][:60]}...")

    # Step 2: Generate natural answer
    print("🤖 Generating answer with LLM...")
    answer = generate_answer(question, context_docs)

    print(f"\n💡 Answer: {answer}")
    return answer

# Test the system
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 RAG CUSTOMER SUPPORT SYSTEM WITH LLM")
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