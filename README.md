# RAG Customer Support API

A production-ready Retrieval-Augmented Generation (RAG) system that answers customer support questions using AI. Built with FastAPI, FAISS vector search, and Groq LLM.

## 🚀 Features

- **Semantic Search**: Finds relevant documents using FAISS vector search
- **AI-Powered Answers**: Generates natural, conversational responses using Groq LLM
- **FastAPI Backend**: REST API with automatic Swagger documentation
- **Docker Support**: Containerized for easy deployment
- **Production Ready**: Environment variables for API keys, logging support

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | Ask a question and get an AI-powered answer |
| GET | `/health` | Check API health status |
| GET | `/documents` | List all documents in the knowledge base |
| GET | `/docs` | Interactive Swagger documentation |

## 🛠️ Tech Stack

- **FastAPI** - Modern Python web framework
- **FAISS** - Facebook's vector similarity search
- **Groq** - LLM API for answer generation
- **Sentence-Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **Docker** - Containerization

## 🏃‍♂️ Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/Aikaksh-Singh-Routela/rag-customer-support-api.git
cd rag-customer-support-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY="your-api-key-here"  # On Windows: set GROQ_API_KEY=your-api-key-here

# Run the API
python main.py
