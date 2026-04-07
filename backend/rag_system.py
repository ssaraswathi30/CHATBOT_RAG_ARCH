import os
import csv
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, csv_folder: str, qdrant_url: str = "http://localhost:6333", qdrant_api_key: str = None, ollama_model: str = "mistral", ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the RAG System
        
        Args:
            csv_folder: Path to folder containing CSV files
            qdrant_url: Qdrant server URL (default: http://localhost:6333)
            qdrant_api_key: Qdrant API key (optional)
            ollama_model: Ollama model to use (default: mistral)
            ollama_base_url: Ollama server base URL (default: http://localhost:11434)
        """
        self.csv_folder = Path(csv_folder)
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = "rag_documents"
        
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {qdrant_url}...")
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Initialize embeddings and LLM
        logger.info("Initializing HuggingFace embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        logger.info(f"Initializing Ollama LLM with model: {ollama_model}")
        self.llm = Ollama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.2
        )
        
        # Load or create vector database
        self.vector_db = None
        self.qa_chain = None
        self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        """Initialize or connect to existing Qdrant database"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Loading existing collection: {self.collection_name}")
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                logger.info(f"Collection has {collection_info.points_count} points")
            else:
                logger.info(f"Collection {self.collection_name} does not exist yet")
                # Create collection with proper vector size (all-MiniLM-L6-v2 uses 384 dimensions)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize Qdrant as LangChain vectorstore
            self.vector_db = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
        except Exception as e:
            logger.error(f"Could not initialize Qdrant database: {str(e)}")
            raise
    
    def load_csvs(self) -> List[Dict[str, Any]]:
        """Load all CSV files from the data folder"""
        documents = []
        
        if not self.csv_folder.exists():
            logger.warning(f"CSV folder does not exist: {self.csv_folder}")
            return documents
        
        csv_files = list(self.csv_folder.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                logger.info(f"Processing {csv_file.name}")
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert row to text document
                        text = " ".join(f"{k}: {v}" for k, v in row.items())
                        documents.append({
                            'text': text,
                            'source': csv_file.name,
                            'metadata': row
                        })
                logger.info(f"Loaded {len([d for d in documents if d['source'] == csv_file.name])} documents from {csv_file.name}")
            except Exception as e:
                logger.error(f"Error reading {csv_file.name}: {str(e)}")
        
        return documents
    
    def sync_csv_to_vector_db(self) -> int:
        """Sync CSV data to vector database"""
        logger.info("Starting CSV to vector database sync...")
        
        # Load documents from CSV
        documents = self.load_csvs()
        
        if not documents:
            logger.warning("No documents loaded from CSV files")
            return 0
        
        # Extract text content - each CSV row becomes one embedding
        texts = [doc['text'] for doc in documents]
        
        logger.info(f"Processing {len(texts)} documents (one embedding per CSV row)")
        
        # Add documents to Qdrant
        try:
            if self.vector_db is None:
                raise Exception("Vector database not initialized")
            
            logger.info("Adding documents to Qdrant...")
            # LangChain Qdrant add_texts returns document IDs
            doc_ids = self.vector_db.add_texts(texts)
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_db.as_retriever(search_kwargs={"k": 3})
            )
            
            logger.info(f"Successfully synced {len(texts)} documents to Qdrant")
            return len(texts)
        except Exception as e:
            logger.error(f"Error syncing to Qdrant: {str(e)}")
            raise
    
    def query(self, user_message: str) -> str:
        """Query the RAG system"""
        if self.qa_chain is None:
            # Try to sync first
            self.sync_csv_to_vector_db()
        
        if self.qa_chain is None:
            return "No documents available. Please sync CSV data first."
        
        try:
            response = self.qa_chain.run(user_message)
            return response
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant vector database"""
        stats = {
            'vector_db_initialized': self.vector_db is not None,
            'documents_count': 0,
            'csv_files': [],
            'qdrant_url': self.qdrant_url
        }
        
        if self.vector_db is not None:
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                stats['documents_count'] = collection_info.points_count
            except Exception as e:
                logger.warning(f"Could not get collection stats: {str(e)}")
                stats['documents_count'] = 0
        
        # Count CSV files
        if self.csv_folder.exists():
            csv_files = list(self.csv_folder.glob("*.csv"))
            stats['csv_files'] = [f.name for f in csv_files]
        
        return stats
