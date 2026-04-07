from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from rag_system import RAGSystem
import logging
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG ChatBot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG System
rag_system = None

class ChatRequest(BaseModel):
    message: str

class SyncResponse(BaseModel):
    status: str
    message: str

class ChatResponse(BaseModel):
    status: str
    response: str

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    csv_folder = os.getenv('CSV_FOLDER', '../data')
    qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
    qdrant_api_key = os.getenv('QDRANT_API_KEY', None)
    ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
    ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    rag_system = RAGSystem(
        csv_folder=csv_folder,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url
    )
    logger.info("RAG System initialized")

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {'status': 'healthy'}

@app.post('/sync-data', response_model=SyncResponse)
async def sync_data():
    """Sync CSV data to vector database"""
    try:
        if rag_system is None:
            raise HTTPException(status_code=500, detail='RAG system not initialized')
        
        result = rag_system.sync_csv_to_vector_db()
        return SyncResponse(
            status='success',
            message=f'Synced {result} documents to vector database'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG"""
    try:
        if rag_system is None:
            raise HTTPException(status_code=500, detail='RAG system not initialized')
        
        if not request.message:
            raise HTTPException(status_code=400, detail='Message is required')
        
        response = rag_system.query(request.message)
        
        return ChatResponse(
            status='success',
            response=response
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/stats')
async def stats():
    """Get vector database stats"""
    try:
        if rag_system is None:
            raise HTTPException(status_code=500, detail='RAG system not initialized')
        
        return rag_system.get_stats()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
