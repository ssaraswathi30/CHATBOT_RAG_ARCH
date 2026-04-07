# RAG ChatBot System

A production-ready Retrieval-Augmented Generation (RAG) chatbot system that syncs CSV data to a vector database and provides intelligent question-answering capabilities using local models.

## 🎯 Features

- **Local Models**: No API costs - uses Ollama for LLM and Sentence Transformers for embeddings
- **Qdrant Vector DB**: High-performance vector database for semantic search
- **CSV Data Sync**: Automatically load and index CSV files into the vector database
- **FastAPI Backend**: Modern, async REST API with automatic documentation
- **Single-Page UI**: Clean, responsive web interface built with vanilla JavaScript
- **Docker & Compose**: Fully containerized with docker-compose orchestration
- **Production Ready**: Health checks, error handling, and comprehensive logging

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FRONTEND (Nginx)                      │
│                   Single Page UI                         │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────────┐
│                   BACKEND (FastAPI)                     │
│              RAG System & REST API                       │
└────────────────────┬────────────────────────────────────┘
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼──┐   ┌───▼────┐  ┌──▼──────┐
    │ Qdrant│   │ Ollama │  │ CSV      │
    │ VecDB │   │  LLM   │  │ Loader   │
    └───────┘   └────────┘  └──────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- At least 8GB RAM (for running Ollama)
- CSV files with log/data content

### 1. Clone/Setup the Project

```bash
cd /path/to/RAG_CHATBOT
```

### 2. Add Your CSV Data

Place your CSV files in the `data/` folder:
```bash
# Example with provided sample
ls data/sample_logs.csv
```

CSV Format Example:
```
timestamp,log_level,service,message,user_id,status_code
2024-03-20 10:15:23,INFO,auth-service,User login successful,user_001,200
```

### 3. Create Environment File

```bash
cd backend
cp .env.example .env
# Edit .env if needed (optional):
# QDRANT_URL=http://localhost:6333
# OLLAMA_MODEL=mistral
```

### 4. Build and Run with Docker Compose

```bash
cd ..
docker-compose up --build
```

This will start:
- **Qdrant Vector DB** → http://localhost:6333
- **Ollama LLM Server** → http://localhost:11434
- **FastAPI Backend** → http://localhost:5000
- **Frontend UI** → http://localhost:80

### 5. Download Ollama Model (First Time Only)

Once Ollama container is running, download the model:

```bash
docker-compose exec ollama ollama pull mistral
# For faster/lighter: ollama pull orca-mini
```

### 6. Access the Interface

Open your browser and navigate to: **http://localhost**

## 📖 API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```
Response: `{"status": "healthy"}`

### Sync CSV Data
```bash
curl -X POST http://localhost:5000/sync-data
```
Response:
```json
{
  "status": "success",
  "message": "Synced 50 documents to vector database"
}
```

### Chat with RAG
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What were the errors in the logs?"}'
```
Response:
```json
{
  "status": "success",
  "response": "Based on the logs, there were 2 errors..."
}
```

### Get Statistics
```bash
curl http://localhost:5000/stats
```
Response:
```json
{
  "vector_db_initialized": true,
  "documents_count": 50,
  "csv_files": ["sample_logs.csv"],
  "qdrant_url": "http://qdrant:6333"
}
```

## 🛠️ Configuration

### Environment Variables

Edit `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CSV_FOLDER` | `/app/data` | Path to CSV files |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | Empty | Optional API key for Qdrant |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL |

### Available Ollama Models

Download different models as needed:
```bash
docker-compose exec ollama ollama pull llama2
docker-compose exec ollama ollama pull neural-chat
docker-compose exec ollama ollama pull orca-mini
```

### Embedding Model

The system uses **all-MiniLM-L6-v2** (384-dimensional embeddings) from Hugging Face. Change in [backend/rag_system.py](backend/rag_system.py#L35):

```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Change this
    model_kwargs={'device': 'cpu'}
)
```

Alternative models:
- `sentence-transformers/all-mpnet-base-v2` (768 dims, slower but better quality)
- `sentence-transformers/all-minilm-l12-v2` (384 dims)

## 📊 System Components

### Backend (`backend/`)
- **app.py**: FastAPI application with REST endpoints
- **rag_system.py**: RAG logic, CSV loading, Qdrant integration
- **requirements.txt**: Python dependencies

### Frontend (`frontend/`)
- **index.html**: Single-page UI with chat interface
- Responsive design with gradient styling
- Real-time synchronization with backend

### Configuration
- **docker-compose.yml**: Service orchestration
- **Dockerfile**: Backend container image
- **nginx.conf**: Reverse proxy configuration

### Data
- **data/**: CSV files for indexing
- **sample_logs.csv**: Example data with log entries

## 🔧 Common Tasks

### Add More CSV Files

1. Place CSV files in `data/` folder
2. Click "Sync Data" button in UI or call `/sync-data` endpoint

### Change LLM Model

Edit `docker-compose.yml` or `.env`:
```bash
OLLAMA_MODEL=llama2
```

Then sync and restart:
```bash
docker-compose restart backend
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f qdrant
docker-compose logs -f ollama
```

### Stop Services

```bash
docker-compose down

# Remove all volumes (clean slate)
docker-compose down -v
```

## 📈 Performance Tips

1. **Chunk Size**: Adjust in [rag_system.py](backend/rag_system.py#L115):
   ```python
   text_splitter = CharacterTextSplitter(
       chunk_size=500,     # Increase for better context
       chunk_overlap=50    # Increase for more overlap
   )
   ```

2. **Retrieval K**: Change number of retrieved documents:
   ```python
   retriever=self.vector_db.as_retriever(search_kwargs={"k": 3})
   ```

3. **Model Quality vs Speed**:
   - Fast: `orca-mini` (3GB)
   - Balanced: `mistral` (7GB) - default
   - Quality: `llama2` (7GB)

## 🐛 Troubleshooting

### Ollama Model Not Found
```bash
docker-compose exec ollama ollama pull mistral
```

### Qdrant Connection Error
```bash
# Check Qdrant is running
curl http://localhost:6333/health

# Restart if needed
docker-compose restart qdrant
```

### High Memory Usage
- Use smaller models: `orca-mini`, `neural-chat`
- Reduce `chunk_size` in rag_system.py
- Increase swap if available

### Slow Inference
- First request takes time (model loading)
- Subsequent requests are faster
- Use lighter models on slower hardware

## 📚 Project Structure

```
RAG_CHATBOT/
├── backend/
│   ├── app.py                 # FastAPI application
│   ├── rag_system.py          # RAG implementation
│   ├── requirements.txt       # Python dependencies
│   └── .env.example          # Environment template
├── frontend/
│   └── index.html            # Single page UI
├── data/
│   └── sample_logs.csv       # Example CSV data
├── docker-compose.yml        # Service orchestration
├── Dockerfile                # Backend container
├── nginx.conf                # Frontend reverse proxy
└── README.md                 # This file
```

## 🔐 Security Considerations

For production deployment:

1. **Qdrant API Key**: Set `QDRANT_API_KEY` environment variable
2. **CORS**: Restrict origins in [app.py](backend/app.py#L17)
3. **HTTPS**: Configure nginx with SSL certificates
4. **Input Validation**: Add request size limits
5. **Rate Limiting**: Use FastAPI rate limiting middleware

## 📝 Example Queries

After syncing sample logs:

- "What services had errors?"
- "List all INFO level messages"
- "Which user had authentication failures?"
- "What was the status of the database backup?"
- "Show me warnings about latency"

## 🤝 Contributing

To extend this system:

1. Add custom RAG chains in `rag_system.py`
2. Modify UI in `frontend/index.html`
3. Update API endpoints in `backend/app.py`
4. Add new CSV processors/loaders

## 📄 License

This project is open source and available for educational and commercial use.

## 🙋 Support

For issues or questions:
1. Check troubleshooting section
2. Review logs: `docker-compose logs -f`
3. Verify services: `docker-compose ps`
4. Check environment variables: `cat backend/.env`

---

**Happy RAG chatbotting!** 🚀
- `orca-mini` (3B) - Lightweight, faster
- `llama2` (7B) - Good reasoning
- `dolphin-mixtral` (45B) - Powerful but slower

To use a different model:

```bash
# Update .env
OLLAMA_MODEL=neural-chat

# Pull the model (if using Docker, run in ollama container)
docker exec rag_chatbot-ollama-1 ollama pull neural-chat

# Restart backend
docker restart rag_chatbot-backend-1
```

## 📊 Usage Example

1. **Place CSV in data folder**:
   ```bash
   cp system_logs.csv data/
   ```

2. **Open UI** and click "Sync Data" button or call:
   ```bash
   curl -X POST http://localhost:5000/sync-data
   ```

3. **Ask questions**:
   - "What are the most common errors?"
   - "Show me all warnings from the auth service"
   - "How many successful logins were there?"

## 🐛 Troubleshooting

### "Ollama connection refused"
- Ensure Ollama service is running
- Check `OLLAMA_BASE_URL` in `.env`
- Try: `docker-compose up ollama`

### "Model not found"
- Pull the model: `docker exec <container-name> ollama pull mistral`
- Or change model in `.env` to one that exists

### "Low memory" or "CUDA out of memory"
- Use smaller model: `OLLAMA_MODEL=orca-mini`
- Ensure Docker has enough memory (Settings > Resources)

### Slow responses
- First query is slower as it loads the model
- Subsequent queries should be faster
- Consider using `orca-mini` for speed trade-off

### CSV not loading
- Verify CSV format with headers
- Check file encoding (UTF-8 recommended)
- Logs appear in backend container: `docker logs <backend-container>`

## 📈 Performance Tips

1. **Chunk Size**: Adjust in `rag_system.py` line 94 (default: 500 chars)
2. **Retrieval Count**: Adjust in `rag_system.py` line 116 (default: k=3)
3. **Model Temperature**: Adjust in `rag_system.py` line 27 (0.0 = deterministic)

## 🛑 Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (careful!)
docker-compose down -v

# Stop specific service
docker-compose stop backend
```

## 📝 Notes

- Vector database persists in `vector_db/` folder
- CSV data is not deleted, only indexed
- First Ollama run downloads ~5GB model
- Embeddings are generated locally (no external API calls)

## 🎯 Next Steps

- Customize sample CSV with your own data
- Adjust Ollama model for better quality/speed trade-off
- Modify frontend styling in `frontend/index.html`
- Add authentication to FastAPI endpoints
- Deploy to cloud (AWS, GCP, Azure)

## 📄 License

MIT License

---

**Happy Chatting!** 🚀
