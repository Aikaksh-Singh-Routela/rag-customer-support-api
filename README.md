# RAG Customer Support API

Production-ready RAG API with FastAPI, FAISS vector search, and Groq Llama 3.3 70B.

## Features

- 🔍 **FAISS Vector Search** - 79.9% retrieval accuracy
- 🤖 **Groq Llama 3.3 70B** - Natural language responses
- 🐳 **Docker Containerized** - One-command deployment
- 📚 **Interactive Swagger Docs** - Test at `/docs`

## Quick Start

### Docker (Recommended)

```bash
docker pull aikaksh/rag-customer-support-api
docker run -p 8000:8000 -e GROQ_API_KEY="your-key" aikaksh/rag-customer-support-api
