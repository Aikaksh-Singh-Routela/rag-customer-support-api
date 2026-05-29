"""
Minimal RAG (Retrieval-Augmented Generation) System
Building this step by step - no course needed!
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample customer support documents (our "knowledge base")
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

print(f"📚 Loaded {len(documents)} documents\n")

# Step 1: Create embeddings (convert text to numbers)
print("🔄 Creating embeddings with sentence-transformers...")
print("   (This downloads a small model once - about 80MB)")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, free model
document_embeddings = model.encode(documents)
print(f"✅ Created {len(document_embeddings)} embeddings")
print(f"   Each embedding has {document_embeddings.shape[1]} dimensions\n")

# Step 2: Create FAISS index for fast similarity search
print("🔍 Building FAISS vector search index...")
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
index.add(document_embeddings.astype('float32'))
print(f"✅ FAISS index created with {index.ntotal} vectors\n")

# Step 3: Search function
def search(query, k=3):
    """Find the k most relevant documents for a query"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    return indices[0], distances[0]

# Step 4: Answer function
def answer_question(question):
    print(f"\n📝 Question: {question}")
    
    # Search for relevant documents
    indices, distances = search(question)
    
    print(f"\n📖 Found {len(indices)} relevant documents:")
    for i, idx in enumerate(indices):
        score = 1 / (1 + distances[i])  # Convert distance to similarity (0-1)
        print(f"   {i+1}. [{score:.2%} similar] {documents[idx][:80]}...")
    
    # Return the best match
    return documents[indices[0]]

# Test the RAG system
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 RAG CUSTOMER SUPPORT SYSTEM - TEST PHASE")
    print("=" * 60)
    
    test_questions = [
        "How do I return something?",
        "What's your return policy?",
        "Is shipping free?",
        "How do I contact support?",
        "What's the warranty on electronics?",
        "Can I track my order?"
    ]
    
    for q in test_questions:
        best_answer = answer_question(q)
        print(f"\n💡 Best answer: {best_answer}\n")
        print("-" * 60)
    
    print("\n✨ RAG system is working! Next step: Add an LLM to generate natural answers.")