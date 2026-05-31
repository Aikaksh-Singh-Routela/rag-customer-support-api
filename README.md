# 🤖 RAG Customer Support API

[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46C3C9?logo=render)](https://aikaksh-rag-customer-support-api.onrender.com)
[![FastAPI](https://img.shields.io/badge/Powered%20by-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-FF6B6B)](https://groq.com)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker)](https://docker.com)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-0052CC)](https://github.com/facebookresearch/faiss)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Logging](https://img.shields.io/badge/Logging-Loguru-FFB300)](https://github.com/Delgan/loguru)


Production-ready RAG (Retrieval-Augmented Generation) API with **FAISS vector search**, **Groq Llama 3.3 70B LLM**, **Docker containerization**, and **full production monitoring**.

---

## 🌐 Live API

**Base URL:** https://aikaksh-rag-customer-support-api.onrender.com

| Endpoint | Method | Description | Link |
|----------|--------|-------------|------|
| `/` | GET | API information | [Try it](https://aikaksh-rag-customer-support-api.onrender.com/) |
| `/docs` | GET | Interactive Swagger UI | [Try it](https://aikaksh-rag-customer-support-api.onrender.com/docs) |
| `/health` | GET/HEAD | Health check for monitoring | [Try it](https://aikaksh-rag-customer-support-api.onrender.com/health) |
| `/metrics` | GET | API metrics (requests, errors, confidence) | [Try it](https://aikaksh-rag-customer-support-api.onrender.com/metrics) |
| `/metrics/detailed` | GET | Detailed metrics dashboard | [Try it](https://aikaksh-rag-customer-support-api.onrender.com/metrics/detailed) |
| `/ask` | POST | Ask a question | [Try it](https://aikaksh-rag-customer-support-api.onrender.com/docs#/default/ask_question_ask_post) |

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🔍 **FAISS Vector Search** | 79.9% retrieval accuracy on customer queries |
| 🤖 **Groq Llama 3.3 70B** | Natural, conversational answers |
| 📊 **Real-time Metrics** | Track requests, errors, confidence scores |
| 🏥 **Health Checks** | Production monitoring ready |
| 🐳 **Docker Containerized** | One-command deployment |
| 📝 **Loguru Logging** | Daily rotation, 7-day retention |
| 🔗 **CORS Enabled** | Ready for frontend integration |
| 📚 **Swagger Docs** | Interactive API documentation |

---

## 📈 Monitoring Dashboard

The API exposes real-time metrics:

```bash
# Health check
curl https://aikaksh-rag-customer-support-api.onrender.com/health

# Basic metrics
curl https://aikaksh-rag-customer-support-api.onrender.com/metrics

# Detailed analytics
curl https://aikaksh-rag-customer-support-api.onrender.com/metrics/detailed
