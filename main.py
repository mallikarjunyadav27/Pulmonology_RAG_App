from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import new RAG architecture
try:
    from rag_architecture import TwoStoreRAGManager, TFIDFLexicalGate
    RAG_ARCHITECTURE_AVAILABLE = True
except ImportError:
    print("RAG architecture not available. Install dependencies: pip install scikit-learn")
    RAG_ARCHITECTURE_AVAILABLE = False
from dotenv import load_dotenv
import os
import json
import io
from datetime import datetime
import whisper
import tempfile
import fitz  # PyMuPDF
import pdfplumber
import re
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from typing import List
import os
import glob
import traceback
from apify_client import ApifyClient
import whisper, torch
import psycopg
import asyncio
import threading

# Import new utilities
try:
    import importlib
    import azure_storage
    importlib.reload(azure_storage)  # Force reload the module
    from azure_storage import get_storage_manager
    AZURE_AVAILABLE = True
except ImportError:
    print("Azure storage not available. Install with: pip install azure-storage-blob")
    AZURE_AVAILABLE = False

try:
    from voice_diarization import get_diarization_processor
    DIARIZATION_AVAILABLE = True
except ImportError:
    print("Voice diarization not available. Install dependencies: pip install pyannote.audio torch")
    DIARIZATION_AVAILABLE = False

# Import Integrated RAG System
try:
    from integrated_rag import IntegratedMedicalRAG
    INTEGRATED_RAG_AVAILABLE = True
    print("✅ Integrated RAG system loaded successfully")
except ImportError:
    print("⚠️ Integrated RAG system not available. Some advanced features may be limited.")
    INTEGRATED_RAG_AVAILABLE = False

from psycopg import sql




BASE_STORAGE_PATH = './KB/'
VECTOR_DB_PATH = './vector_dbs/'
ORGANIZATION_KB_PATH = './Organization_KB/'
ORGANIZATION_VECTOR_DB_PATH = './vector_dbs/organization/'

# Create required directories
os.makedirs(BASE_STORAGE_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(ORGANIZATION_KB_PATH, exist_ok=True)
os.makedirs(ORGANIZATION_VECTOR_DB_PATH, exist_ok=True)

last_created_folder = None 
VECTOR_DBS_FOLDER = "./vector_dbs"

def get_timestamp():
    """Generate timestamp in MMDDYYYYHHMM format."""
    return time.strftime("%m%d%Y%H%M")

def get_latest_vector_db():
    """Finds the latest vector database in the vector_dbs folder."""
    vector_dbs = glob.glob(os.path.join(VECTOR_DBS_FOLDER, "*"))  # Adjust if needed

    if not vector_dbs:
        print("No existing vector DB found. A new one will be created.")
        return None

    latest_db = max(vector_dbs, key=os.path.getmtime)
    print(f"Using latest vector DB: {latest_db}")
    return latest_db

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

# Load environment variables
load_dotenv()

# Database configuration
db_config = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
}

# ---------------------------------------------------------------------------
# PostgreSQL connection pool — reuses connections across requests so each
# endpoint doesn't pay the full TCP handshake cost to the remote Azure host.
# Falls back to per-request connections if psycopg_pool is not installed.
# ---------------------------------------------------------------------------
try:
    from psycopg_pool import ConnectionPool as _PGPool
    from contextlib import contextmanager as _contextmanager

    _pool_kwargs = {k: v for k, v in db_config.items() if v is not None}
    _pg_pool = _PGPool(
        conninfo="",
        kwargs=_pool_kwargs,
        min_size=1,
        max_size=10,
        open=False,
        reconnect_timeout=30,
    )
    _pg_pool.open(wait=False)  # Background open — doesn't block startup

    @_contextmanager
    def _pg_conn():
        with _pg_pool.connection() as conn:
            yield conn

    print("✅ PostgreSQL connection pool initialised (psycopg_pool)")
except ImportError:
    from contextlib import contextmanager as _contextmanager

    _pg_pool = None

    @_contextmanager
    def _pg_conn():
        with psycopg.connect(**db_config) as conn:
            yield conn

    print("⚠️  psycopg_pool not installed — using per-request connections. "
          "Run: pip install 'psycopg[pool]'")

# Initialize Flask app
app = Flask(__name__)

persist_directory = "./vector_db"


# Initialize LLM
llm = ChatOpenAI(
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("base_url"),  # https://api.openai.com/v1
    model_name=os.getenv("llm_model_name"),  # gpt-4o-mini
    request_timeout=30  # Add 30 second timeout
)

def create_contextual_llm(patient_context: str = None) -> ChatOpenAI:
    """
    Create an LLM instance with optional patient context as system message.
    
    Args:
        patient_context: Patient context to be used as system message
        
    Returns:
        ChatOpenAI instance configured with patient context
    """
    base_system_message = "You are a medical AI assistant providing accurate, evidence-based medical information and guidance."
    
    if patient_context:
        system_message = f"Patient Context: {patient_context}\n\n{base_system_message} Always consider the patient context when providing medical advice and recommendations. Tailor your responses to the specific patient demographics, conditions, and medical history provided."
    else:
        system_message = base_system_message
    
    # Create LLM with system message
    contextual_llm = ChatOpenAI(
        api_key=os.getenv("openai_api_key"),
        base_url=os.getenv("base_url"),
        model_name=os.getenv("llm_model_name"),
        temperature=0.1,  # Lower temperature for medical advice
        request_timeout=30  # Add 30 second timeout
    )
    
    # Store system message for use in chains
    contextual_llm._system_message = system_message
    
    return contextual_llm

client = ApifyClient(os.getenv("apify_api_key"))  # Initialize Apify client

# Load disciplines configuration
def load_disciplines_config():
    """Load disciplines configuration from JSON file."""
    try:
        with open("config/disciplines.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Warning: disciplines.json not found. Using default configuration.")
        return {
            "disciplines": [
                {
                    "id": "family_medicine",
                    "name": "Family Medicine", 
                    "description": "Comprehensive primary healthcare",
                    "is_default": True,
                    "kb_path": "Organization_KB/Family_Medicine",
                    "vector_db_path": "vector_dbs/organization/family_medicine"
                }
            ],
            "selection_rules": {
                "min_selections": 1,
                "max_selections": 3,
                "default_discipline": "family_medicine"
            }
        }

# Load configuration
disciplines_config = load_disciplines_config()

class MedicalQueryRouter:
    """Intelligent router that determines which medical disciplines are relevant for a query."""
    
    def __init__(self, llm, disciplines_config):
        self.llm = llm
        self.disciplines = disciplines_config.get("disciplines", [])
        self.discipline_keywords = self._build_keyword_map()
    
    def _build_keyword_map(self):
        """Build a map of keywords for each discipline."""
        keyword_map = {}
        
        # Medical specialty keywords
        specialty_keywords = {
            "family_medicine": [
                "primary care", "general practice", "family doctor", "annual checkup", "preventive care",
                "common cold", "flu", "hypertension", "diabetes", "vaccination", "routine care",
                "wellness exam", "physical exam", "blood pressure", "cholesterol", "general health"
            ],
            "cardiology": [
                "heart", "cardiac", "cardiovascular", "chest pain", "heart attack", "myocardial infarction",
                "heart failure", "arrhythmia", "atrial fibrillation", "coronary", "angina", "pacemaker",
                "cardiologist", "EKG", "ECG", "echocardiogram", "blood pressure", "hypertension",
                "heart rate", "cardiac arrest", "valve", "aorta", "coronary artery"
            ],
            "neurology": [
                "brain", "neurological", "nervous system", "stroke", "seizure", "epilepsy", "migraine",
                "headache", "Parkinson's", "Alzheimer's", "dementia", "multiple sclerosis", "MS",
                "neurologist", "MRI brain", "CT brain", "memory loss", "confusion", "dizziness",
                "numbness", "tingling", "weakness", "paralysis", "spinal cord", "nerve"
            ],
            "doctors_files": [
                "my files", "my documents", "uploaded", "document", "file", "PDF", "article",
                "my upload", "personal documents", "doctor's files", "my records", "uploaded content",
                "session files", "my PDFs", "document I uploaded", "file I shared", "my data"
            ]
        }
        
        return specialty_keywords
    
    def _has_session_files(self):
        """Check if the current session has uploaded files."""
        global last_created_folder
        if not last_created_folder:
            return False
        
        # Check for PDFs in session
        pdf_path = os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder)
        url_path = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)
        
        pdf_files = []
        url_files = []
        
        if os.path.exists(pdf_path):
            pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
        
        if os.path.exists(url_path):
            url_files = [f for f in os.listdir(url_path) if f.endswith('.txt')]
        
        return len(pdf_files) > 0 or len(url_files) > 0
    
    def analyze_query(self, query):
        """Analyze query and determine relevant disciplines using AI + keywords."""
        query_lower = query.lower()
        
        # First, use keyword matching for quick routing
        relevant_disciplines = []
        confidence_scores = {}
        
        for discipline_id, keywords in self.discipline_keywords.items():
            keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
            if keyword_matches > 0:
                confidence = min(keyword_matches / len(keywords) * 100, 95)  # Cap at 95%
                relevant_disciplines.append(discipline_id)
                confidence_scores[discipline_id] = confidence
        
        # Special handling for doctors_files - include if user has uploaded files and query might be relevant
        has_files = self._has_session_files()
        if has_files and "doctors_files" not in relevant_disciplines:
            # Check if query could be asking about user's files (more lenient keywords)
            user_file_keywords = ["my", "document", "file", "upload", "PDF", "article", "personal", "doctor", "record"]
            if any(keyword in query_lower for keyword in user_file_keywords):
                relevant_disciplines.append("doctors_files")
                confidence_scores["doctors_files"] = 85  # High confidence for user file queries
        
        # If no keyword matches, use AI to analyze
        if not relevant_disciplines:
            relevant_disciplines = self._ai_analyze_query(query)
            for discipline in relevant_disciplines:
                confidence_scores[discipline] = 70  # Default AI confidence
            
            # Add doctors_files to AI analysis if user has files
            if has_files and "doctors_files" not in relevant_disciplines:
                relevant_disciplines.append("doctors_files")
                confidence_scores["doctors_files"] = 75
        
        # Ensure we have at least one discipline (default to family medicine)
        if not relevant_disciplines:
            relevant_disciplines = ["family_medicine"]
            confidence_scores["family_medicine"] = 60
        
        # Sort by confidence
        relevant_disciplines.sort(key=lambda d: confidence_scores.get(d, 0), reverse=True)
        
        return {
            "disciplines": relevant_disciplines[:2],  # Limit to top 3
            "confidence_scores": confidence_scores,
            "routing_method": "hybrid" if len(relevant_disciplines) > 0 else "default"
        }
    
    def _ai_analyze_query(self, query):
        """Use AI to analyze query when keyword matching fails."""
        try:
            discipline_names = [d["name"] for d in self.disciplines]
            
            prompt = f"""
            Analyze this medical query and determine which medical specialties are most relevant:
            
            Query: "{query}"
            
            Available specialties: {', '.join(discipline_names)}
            
            Guidelines:
            - If the query mentions "my files", "my documents", "uploaded", or refers to user's personal documents, include "Doctor's Files"
            - If the query is general or could apply to multiple specialties, include Family Medicine
            - If unclear, default to Family Medicine
            - Consider that "Doctor's Files" contains user-uploaded PDFs and documents
            
            Return only the specialty names that are relevant, separated by commas.
            Response format: Specialty1, Specialty2 (max 3)
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse AI response and map to discipline IDs
            ai_specialties = [s.strip() for s in content.split(',')]
            relevant_disciplines = []
            
            for specialty in ai_specialties:
                for discipline in self.disciplines:
                    if discipline["name"].lower() in specialty.lower():
                        relevant_disciplines.append(discipline["id"])
                        break
            
            return relevant_disciplines
            
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return ["family_medicine"]  # Fallback

# Initialize the router
medical_router = MedicalQueryRouter(llm, disciplines_config)

def get_available_disciplines():
    """Return list of available disciplines for UI dropdown."""
    return disciplines_config.get("disciplines", [])

def validate_discipline_selection(selected_disciplines):
    """Validate user's discipline selection against rules."""
    rules = disciplines_config.get("selection_rules", {})
    min_sel = rules.get("min_selections", 1)
    max_sel = rules.get("max_selections", 3)
    
    if len(selected_disciplines) < min_sel:
        return False, f"Please select at least {min_sel} discipline(s)"
    if len(selected_disciplines) > max_sel:
        return False, f"Please select no more than {max_sel} discipline(s)"
    
    # Validate discipline IDs exist
    valid_ids = [d["id"] for d in disciplines_config.get("disciplines", [])]
    invalid_ids = [d for d in selected_disciplines if d not in valid_ids]
    if invalid_ids:
        return False, f"Invalid discipline(s): {', '.join(invalid_ids)}"
    
    return True, "Valid selection"

def get_discipline_vector_db_path(discipline_id):
    """Get vector database path for a specific discipline."""
    for discipline in disciplines_config.get("disciplines", []):
        if discipline["id"] == discipline_id:
            return discipline.get("vector_db_path", "")
    return None

def create_organization_vector_db(discipline_id, documents):
    """Create or update organization vector database for a specific discipline."""
    vector_db_path = get_discipline_vector_db_path(discipline_id)
    if not vector_db_path:
        raise ValueError(f"Unknown discipline: {discipline_id}")
    
    persist_dir = os.path.join(".", vector_db_path)
    os.makedirs(persist_dir, exist_ok=True)
    
    # Create or update the vector store
    vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    return vector_store

# Step 1: Load and Process Metadata
def load_metadata(file_path: str) -> List[dict]:
    """Load JSON metadata from the given file path."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

def process_pdf_metadata(pdf_metadata: list, text_splitter) -> list:
    """Chunk the text content from PDF metadata."""
    chunked_documents = []

    for doc in pdf_metadata:
        file_name = doc.get("file_name", "Unknown File")
        pages = doc.get("text", [])
        
        for page in pages:
            page_number = page.get("page", "Unknown Page")
            page_text = page.get("text", "").strip()

            # Skip empty pages
            if not page_text:
                continue

            # Split text and create chunks
            chunks = text_splitter.split_text(page_text)
            for chunk in chunks:
                chunked_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": file_name, "type": "pdf", "page": page_number}
                    )
                )

    return chunked_documents

import json

def process_url_metadata(url_metadata: list, text_splitter) -> list:
    """Chunk the text content from URL metadata."""
    chunked_documents = []

    for entry in url_metadata:
        url = entry.get("url", "Unknown URL")
        text_content = entry.get("text", "").strip()
        date_info = entry.get("date", "Unknown Date")

        # Skip entries with empty text
        if not text_content:
            continue

        # Split the content and create document chunks
        chunks = text_splitter.split_text(text_content)
        for chunk in chunks:
            chunked_documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": url, "type": "url", "date": date_info}
                )
            )

    return chunked_documents


# Load and process metadata
pdf_metadata = load_metadata("pdf_metadata.json")
url_metadata = load_metadata("url_metadata.json")


# Create embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.getenv('openai_api_key'),
    base_url=os.getenv('base_url'),
    model=os.getenv('embedding_model_name')
)

# Initialize Two-Store RAG Manager
rag_manager = None
if RAG_ARCHITECTURE_AVAILABLE:
    try:
        rag_manager = TwoStoreRAGManager(embeddings, llm, VECTOR_DBS_FOLDER)
        print("✅ Two-Store RAG Manager initialized successfully")
        
        # Check if external KB already has content to avoid reloading
        kb_external_has_content = False
        if rag_manager.kb_external:
            try:
                # Check if external KB has documents
                count = rag_manager.kb_external._collection.count()
                kb_external_has_content = count > 0
                print(f"📊 External KB already contains {count} documents")
            except:
                kb_external_has_content = False
        
        # Only load external content if KB is empty
        if not kb_external_has_content:
            print("🌐 External KB is empty, loading initial content...")
            
            # Initialize external knowledge base with some medical topics
            medical_topics = [
                "pulmonology", "cardiology", "neurology", "family medicine", 
                "medical diagnosis", "clinical medicine", "pharmacology"
            ]
            print(f"📚 Loading {len(medical_topics)} Wikipedia topics...")
            rag_manager.load_wikipedia_content(medical_topics, max_docs_per_topic=2)
            
            # Load some medical research from arXiv
            arxiv_queries = [
                "medical diagnosis AI", "clinical decision support", 
                "medical imaging analysis", "healthcare machine learning"
            ]
            print(f"🔬 Loading {len(arxiv_queries)} arXiv queries...")
            rag_manager.load_arxiv_content(arxiv_queries, max_docs_per_query=1)
            
            print("✅ External KB initial content loaded successfully")
        else:
            print("✅ External KB already populated, skipping content loading")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG Manager: {e}")
        rag_manager = None
else:
    print("⚠️ RAG Architecture not available - using legacy mode")

# Initialize Integrated Medical RAG System
integrated_rag_system = None
if INTEGRATED_RAG_AVAILABLE:
    try:
        api_key = os.getenv('openai_api_key')
        if api_key:
            integrated_rag_system = IntegratedMedicalRAG(
                openai_api_key=api_key,
                base_vector_path=VECTOR_DBS_FOLDER
            )
            print("✅ Integrated Medical RAG System initialized successfully")
        else:
            print("⚠️ OpenAI API key not found - Integrated RAG system disabled")
    except Exception as e:
        print(f"❌ Failed to initialize Integrated RAG System: {e}")
        integrated_rag_system = None
else:
    print("⚠️ Integrated RAG System not available")

# Split documents and create FAISS vector store
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4096, 
        chunk_overlap=128,
        separators=["\n\n", "\n", ".", " "]
    )

pdf_documents = process_pdf_metadata(pdf_metadata, text_splitter)
url_documents = process_url_metadata(url_metadata,text_splitter)
all_documents = pdf_documents + url_documents


# if os.path.exists(persist_directory) and os.listdir(persist_directory):
#     print(f"Persist directory '{persist_directory}' found. Skipping embedding.")
#     vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# else:
#     print("Persist directory not found. Creating embeddings and initializing Chroma...")
#     vector_store = Chroma.from_documents(all_documents, embedding=embeddings, persist_directory=persist_directory)
    


# # Create retriever and QA chain
# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Helper function to enhance response with citations
def enhance_with_citations(results):
    pdf_citations = set()  # Track unique PDF citations
    url_citations = set()  # Track unique URL citations
    org_citations = set()  # Track unique Organization KB citations

    for doc in results:
        metadata = getattr(doc, "metadata", {})  # Ensure metadata exists
        doc_type = metadata.get("type")  # Check for 'type'

        if doc_type == "pdf":
            pdf_source = metadata.get("source", "Unknown PDF")
            page_info = metadata.get("page", "Unknown Page")
            pdf_citations.add(f"PDF: {pdf_source} (Page {page_info})")

        elif doc_type == "url":
            url_source = metadata.get("source", "Unknown URL")
            url_citations.add(f"URL: {url_source}")
            
        elif doc_type == "organization_pdf":
            org_source = metadata.get("source", "Unknown Document")
            discipline = metadata.get("discipline", "Unknown Discipline")
            page_info = metadata.get("page", "Unknown Page")
            org_citations.add(f"Organization KB - {discipline.replace('_', ' ').title()}: {org_source} (Page {page_info})")

    # Combine citations
    all_citations = pdf_citations.union(url_citations).union(org_citations)
    return "\n".join(all_citations) or "No citations available"

def clean_extracted_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!?\'":;\-\s]', '', text)
    text = text.strip()
    text = re.sub(r'\.{2,}', '.', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

def extract_text_from_pdf(pdf_file):
    text_content = []
    with fitz.open(pdf_file) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            clean_text = clean_extracted_text(page.get_text())
            text_content.append(clean_text)
    return text_content

# def extract_text_from_url(url):
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("start-maximized")
#     chrome_options.add_argument("--disable-blink-features=AutomationControlled")
#     chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    
#     service = Service(executable_path="/usr/local/bin/chromedriver")
#     driver = webdriver.Chrome(service=service, options=chrome_options)
    
#     try:
#         driver.get(url)
#         html_content = driver.page_source
#         soup = BeautifulSoup(html_content, 'html.parser')
#         for script_or_style in soup(["script", "style"]):
#             script_or_style.decompose()
#         text = soup.get_text(separator=" ")
#         text = re.sub(r"[^\x00-\x7F]+", " ", text)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text
#     finally:
#         driver.quit()



def extract_text_from_url(url):
    # Define run input with Playwright crawler and filtering
    run_input = {
        "startUrls": [{"url": url}],
        "useSitemaps": False,
        "respectRobotsTxtFile": True,
        "crawlerType": "playwright:adaptive",
        "includeUrlGlobs": [],
        "excludeUrlGlobs": [],
        "initialCookies": [],
        "proxyConfiguration": {"useApifyProxy": True},
        "keepElementsCssSelector": "",
        "removeElementsCssSelector": """nav, footer, script, style, noscript, svg, img[src^='data:'],
        [role=\"alert\"],
        [role=\"banner\"],
        [role=\"dialog\"],
        [role=\"alertdialog\"],
        [role=\"region\"][aria-label*=\"skip\" i],
        [aria-modal=\"true\"]""",
        "clickElementsCssSelector": "[aria-expanded=\"false\"]",
            }

    # Run the Apify actor
    run = client.actor("apify/website-content-crawler").call(run_input=run_input)
    
    # Collect and clean text content
    full_text = ""
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        page_text = item.get("text", "")
        page_text = re.sub(r"[^\x00-\x7F]+", " ", page_text)  # Remove non-ASCII
        page_text = re.sub(r"\s+", " ", page_text).strip()    # Normalize whitespace
        full_text += page_text + "\n"
    print(full_text)
    return full_text.strip()



@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_file.save(temp.name)
        result = model.transcribe(temp.name)
    return jsonify({"text": result['text']})

@app.route('/translate_audio', methods=['POST'])
def translate_audio():
    """
    Transcribe audio in selected language, identify speakers (doctor/patient), and translate to English.
    OPTIMIZED: Combined segmentation + translation in single LLM call
    Expects: audio file, language code
    Returns: segmented conversation with speaker roles and translations
    """
    from langchain.schema import HumanMessage, SystemMessage
    import concurrent.futures
    
    try:
        audio_file = request.files.get('audio')
        language_code = request.form.get('language', 'es')  # Default to Spanish
        
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400
        
        # Language code mapping for Whisper
        language_map = {
            'es': 'spanish',
            'zh': 'chinese',
            'yue': 'chinese',  # Cantonese (use Chinese for Whisper)
            'tl': 'tagalog',
            'hi': 'hindi',
            'te': 'telugu',
            'ta': 'tamil',
            'gu': 'gujarati',
            'pa': 'punjabi'
        }
        
        whisper_language = language_map.get(language_code, 'spanish')
        
        # Transcribe audio using Whisper with specified language
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                temp_path = temp.name
                audio_file.save(temp_path)
                print(f"Translation audio file saved to: {temp_path}")
                
                # SPEED OPTIMIZATION: Use fp16, no_speech_threshold, and faster beam search
                result = model.transcribe(
                    temp_path, 
                    language=whisper_language,
                    fp16=device == "cuda",  # Use fp16 on GPU for 2x speed
                    beam_size=1,  # Faster beam search (greedy decoding)
                    best_of=1,  # Don't compare multiple samples
                    temperature=0.0,  # Deterministic output
                    condition_on_previous_text=False  # Faster processing
                )
                original_text = result['text']
                print(f"✅ Fast transcription completed in {whisper_language}. Length: {len(original_text)} characters")
                
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # OPTIMIZATION: Combine segmentation and translation in ONE LLM call
        try:
            needs_translation = (language_code != 'en' and whisper_language != 'english')
            
            if needs_translation:
                # Combined prompt for segmentation + translation
                combined_prompt = f"""Analyze this doctor-patient conversation in {whisper_language} and do TWO things:
1. Segment it by speaker (doctor vs patient) - EACH TURN/EXCHANGE should be a SEPARATE segment
2. Translate each segment to English

Transcript:
{original_text}

Provide your response in this exact JSON format:
{{
  "segments": [
    {{"speaker": "doctor", "text": "original text", "translated_text": "English translation"}},
    {{"speaker": "patient", "text": "original text", "translated_text": "English translation"}},
    {{"speaker": "doctor", "text": "original text", "translated_text": "English translation"}},
    {{"speaker": "patient", "text": "original text", "translated_text": "English translation"}}
  ]
}}

CRITICAL Rules:
- Split the conversation into INDIVIDUAL turns/exchanges - do NOT combine all doctor statements into one segment
- Each time the speaker changes, create a NEW segment
- Medical professionals use medical terminology, ask clinical questions, give advice
- Patients describe symptoms, ask questions about their health
- Provide accurate medical translations
- Be precise in segmentation - multiple back-and-forth exchanges should result in multiple segments"""
            else:
                # English - only segmentation needed
                combined_prompt = f"""Analyze this doctor-patient conversation and segment it by speaker.

Transcript:
{original_text}

Provide your response in this exact JSON format:
{{
  "segments": [
    {{"speaker": "doctor", "text": "the text spoken", "translated_text": "the text spoken"}},
    {{"speaker": "patient", "text": "the text spoken", "translated_text": "the text spoken"}}
  ]
}}

Rules:
- Medical professionals use medical terminology, ask clinical questions, give advice
- Patients describe symptoms, ask questions about their health"""
            
            messages = [
                SystemMessage(content="You are an expert medical conversation analyzer and translator. Provide responses in valid JSON format only."),
                HumanMessage(content=combined_prompt)
            ]
            
            # SPEED OPTIMIZATION: Use faster LLM with optimized settings
            from langchain_openai import ChatOpenAI
            fast_llm = ChatOpenAI(
                api_key=os.getenv("openai_api_key"),
                base_url=os.getenv("base_url"),
                model_name=os.getenv("llm_model_name"),
                temperature=0.3,  # Faster inference with slight randomness
                max_tokens=2000,  # Limit output length for speed
                request_timeout=30
            )
            
            response = fast_llm.invoke(messages)
            result_content = response.content.strip()
            
            # Parse the JSON response
            # Extract JSON from markdown code blocks if present
            if "```json" in result_content:
                result_content = result_content.split("```json")[1].split("```")[0].strip()
            elif "```" in result_content:
                result_content = result_content.split("```")[1].split("```")[0].strip()
            
            segments_data = json.loads(result_content)
            segments = segments_data.get('segments', [])
            
            # Ensure all segments have translated_text
            for segment in segments:
                if 'translated_text' not in segment:
                    segment['translated_text'] = segment.get('text', '')
            
            print(f"Completed: {len(segments)} segments identified and translated in single call")
            
            return jsonify({
                "segments": segments,
                "has_segments": True
            })
            
        except json.JSONDecodeError as je:
            print(f"JSON parsing error: {str(je)}")
            print(f"Raw response: {result_content if 'result_content' in locals() else 'N/A'}")
            
            # Fallback: return as single segment
            if needs_translation:
                fallback_prompt = f"""Translate this {whisper_language} medical conversation to English. Provide only the translation.

{original_text}"""
                
                messages = [
                    SystemMessage(content="You are a professional medical translator."),
                    HumanMessage(content=fallback_prompt)
                ]
                
                response = llm.invoke(messages)
                translated_text = response.content.strip()
            else:
                translated_text = original_text
            
            return jsonify({
                "original_text": original_text,
                "translated_text": translated_text,
                "has_segments": False
            })
            
        except Exception as e:
            print(f"Error during segmentation/translation: {str(e)}")
            traceback.print_exc()
            
            # Fallback: return as single segment without speaker identification
            if needs_translation:
                translation_prompt = f"""Translate the following {whisper_language} text to English.
Provide only the English translation.

Original text:
{original_text}"""
                
                messages = [
                    SystemMessage(content="You are a professional medical translator."),
                    HumanMessage(content=translation_prompt)
                ]
                
                response = llm.invoke(messages)
                translated_text = response.content.strip()
            else:
                translated_text = original_text
            
            return jsonify({
                "original_text": original_text,
                "translated_text": translated_text,
                "has_segments": False
            })
    
    except Exception as e:
        print(f"Unexpected error in translate_audio: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe_patient_notes', methods=['POST'])
def transcribe_patient_notes():
    """
    Transcribe patient audio recording and generate medical summary.
    Expects: audio file, doctor_name, patient_name
    Returns: transcribed text, summary, and conclusion
    """
    try:
        # Get form data
        audio_file = request.files.get('audio')
        doctor_name = request.form.get('doctor_name', '')
        patient_name = request.form.get('patient_name', '')
        
        print(f"Processing patient recording for: {patient_name} by Dr. {doctor_name}")
        
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400
        
        if not doctor_name or not patient_name:
            return jsonify({"error": "Doctor name and patient name are required"}), 400
        
        # Transcribe audio using Whisper
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                temp_path = temp.name
                audio_file.save(temp_path)
                print(f"Audio file saved to: {temp_path}")
                
                result = model.transcribe(temp_path)
                transcribed_text = result['text']
                print(f"Transcription completed. Length: {len(transcribed_text)} characters")
                
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Generate medical summary using OpenAI
        summary_prompt = f"""
        As a medical AI assistant, please analyze the following patient consultation transcript and provide a professional medical summary.
        
        Patient: {patient_name}
        Doctor: {doctor_name}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Transcript:
        {transcribed_text}
        
        Please provide:
        1. A concise clinical summary highlighting key medical information, symptoms, findings, and discussions
        2. Professional conclusions with recommendations, follow-up actions, or treatment plans mentioned
        
        Format your response exactly as:
        SUMMARY:
        [Provide a clear, professional summary of the medical consultation]
        
        CONCLUSION:
        [Provide conclusions, recommendations, and any follow-up actions mentioned]
        """
        
        try:
            # Get AI response
            print("Generating medical summary...")
            ai_response = llm.invoke(summary_prompt)
            if hasattr(ai_response, 'content'):
                ai_content = ai_response.content.strip()
            else:
                ai_content = str(ai_response).strip()
            
            print(f"AI summary generated. Length: {len(ai_content)} characters")
            
            # Parse summary and conclusion from AI response
            summary_parts = ai_content.split("CONCLUSION:")
            if len(summary_parts) == 2:
                summary = summary_parts[0].replace("SUMMARY:", "").strip()
                conclusion = summary_parts[1].strip()
            else:
                # Fallback parsing method
                lines = ai_content.split('\n')
                summary_lines = []
                conclusion_lines = []
                in_conclusion = False
                
                for line in lines:
                    if 'CONCLUSION' in line.upper():
                        in_conclusion = True
                        continue
                    elif 'SUMMARY' in line.upper():
                        in_conclusion = False
                        continue
                    
                    if in_conclusion:
                        conclusion_lines.append(line)
                    else:
                        summary_lines.append(line)
                
                summary = '\n'.join(summary_lines).strip()
                conclusion = '\n'.join(conclusion_lines).strip()
                
                # Final fallback
                if not summary and not conclusion:
                    summary = ai_content[:len(ai_content)//2]
                    conclusion = ai_content[len(ai_content)//2:]
        
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500
        
        response_data = {
            "success": True,
            "transcribed_text": transcribed_text,
            "summary": summary,
            "conclusion": conclusion,
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Successfully processed patient recording for {patient_name}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Unexpected error in transcribe_patient_notes: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    """
    Generate medical summary and conclusion from transcription text.
    Expects: transcription text, doctor_name, patient_name
    Returns: summary and conclusion
    """
    try:
        data = request.get_json()
        transcription = data.get('transcription', '').strip()
        doctor_name = data.get('doctor_name', '').strip()
        patient_name = data.get('patient_name', '').strip()
        
        if not transcription:
            return jsonify({"success": False, "error": "No transcription text provided"}), 400
        
        print(f"Generating summary for patient: {patient_name} by Dr. {doctor_name}")
        
        # Generate medical summary using OpenAI
        summary_prompt = f"""
        As a medical AI assistant, please analyze the following patient consultation transcript and provide a professional medical summary.
        
        Patient: {patient_name if patient_name else 'Not specified'}
        Doctor: {doctor_name if doctor_name else 'Not specified'}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Transcript:
        {transcription}
        
        Please provide:
        1. A concise clinical summary highlighting key medical information, symptoms, findings, and discussions
        2. Professional conclusions with recommendations, follow-up actions, or treatment plans mentioned
        
        If the transcript does not contain relevant medical information, please provide appropriate default responses indicating the lack of medical content.
        
        Format your response exactly as:
        SUMMARY:
        [Provide a clear, professional summary of the medical consultation]
        
        CONCLUSION:
        [Provide conclusions, recommendations, and any follow-up actions mentioned]
        """
        
        try:
            # Get AI response
            print("Generating medical summary from transcription...")
            ai_response = llm.invoke(summary_prompt)
            if hasattr(ai_response, 'content'):
                ai_content = ai_response.content.strip()
            else:
                ai_content = str(ai_response).strip()
            
            print(f"AI summary generated. Length: {len(ai_content)} characters")
            
            # Parse summary and conclusion from AI response
            summary_parts = ai_content.split("CONCLUSION:")
            if len(summary_parts) == 2:
                summary = summary_parts[0].replace("SUMMARY:", "").strip()
                conclusion = summary_parts[1].strip()
            else:
                # Fallback parsing method
                lines = ai_content.split('\n')
                summary_lines = []
                conclusion_lines = []
                in_conclusion = False
                
                for line in lines:
                    if 'CONCLUSION' in line.upper():
                        in_conclusion = True
                        continue
                    elif 'SUMMARY' in line.upper():
                        in_conclusion = False
                        continue
                    
                    if in_conclusion:
                        conclusion_lines.append(line)
                    else:
                        summary_lines.append(line)
                
                summary = '\n'.join(summary_lines).strip()
                conclusion = '\n'.join(conclusion_lines).strip()
                
                # Final fallback with appropriate default responses
                if not summary and not conclusion:
                    summary = "The consultation transcript provided does not contain any relevant medical information, symptoms, findings, or discussions related to a patient's health."
                    conclusion = "As there is no pertinent information available in the transcript, no medical conclusions, recommendations, or follow-up actions can be provided. It is recommended to ensure accurate and detailed documentation of patient consultations for proper medical assessment and care."
        
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return jsonify({"success": False, "error": f"Summary generation failed: {str(e)}"}), 500
        
        return jsonify({
            "success": True,
            "summary": summary,
            "conclusion": conclusion
        })
        
    except Exception as e:
        print(f"Unexpected error in generate_summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/plain_english", methods=["POST"])
def plain_english():
    user_text = request.json.get("text", "")
    if not user_text:
        return jsonify({"refined_text": "", "message": "No input provided."})

    try:
        prompt = f"Rewrite the following question in plain English for better clarity:\n\n{user_text}"
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            refined_text = response.content.strip()
        else:
            refined_text = str(response).strip()

        return jsonify({"refined_text": refined_text})
    except Exception as e:
        return jsonify({"refined_text": "", "message": f"Error: {str(e)}"})



def clean_response_text(text):
    """Remove emojis and clean up response text while preserving line breaks."""
    import re
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        u"\U00002700-\U000027BF"  # dingbats
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub('', text)
    
    # Clean up extra spaces but preserve line breaks
    # First, handle multiple spaces within lines
    text = re.sub(r'[ \t]+', ' ', text)
    # Then clean up excessive line breaks (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing spaces at the end of lines
    text = re.sub(r' +\n', '\n', text)
    
    return text.strip()

def generate_full_html_response(result_data):
    """
    ✅ Guaranteed vertical layout version.
    Everything is inside ONE single <div>, so host grid/flex can't split them into columns.
    """

    # Extract data
    medical_summary = result_data.get("medical_summary", "No medical summary available.")
    sources = result_data.get("sources", [])
    tool_info = result_data.get("tool_info", {})
    primary_tool = tool_info.get("primary_tool", "Unknown")
    confidence = tool_info.get("confidence", "Unknown")
    tools_used = tool_info.get("tools_used", "N/A")
    reasoning = tool_info.get("reasoning", "No reasoning provided.")

    # Format sources
    sources_html = (
        "<ul>" + "".join(f"<li>{s}</li>" for s in sources) + "</ul>"
        if sources else "<p>No sources available.</p>"
    )

    # ✅ Everything inside ONE single <div>
    html = f"""
<div style="display:block; width:100%; max-width:100%; line-height:1.6;">
  <h3 style="color:#007bff; font-size:20px; margin-bottom:10px;">📋 Medical Summary</h3>
  <div style="background:#e3f2fd; padding:15px; border-radius:8px; margin-bottom:30px;">
    {medical_summary}
  </div>

  <h3 style="color:#6f42c1; font-size:20px; margin-bottom:10px;">📖 Sources</h3>
  <div style="background:#f8f9fa; padding:15px; border-radius:8px; margin-bottom:30px;">
    {sources_html}
  </div>

  <h3 style="color:#ff6600; font-size:20px; margin-bottom:10px;">🔧 Tool Selection &amp; Query Routing</h3>
  <div style="background:#fff3cd; padding:15px; border-radius:8px;">
    <p><strong>Primary Tool:</strong> {primary_tool}</p>
    <p><strong>Confidence:</strong> {confidence}</p>
    <p><strong>Tools Used:</strong> {tools_used}</p>
    <p><strong>Reasoning:</strong> {reasoning}</p>
  </div>
</div>
"""
    return html


def parse_enhanced_response(answer, routing_info, tools_used, explanation):
    """Parse enhanced HTML response and extract structured data for new HTML format"""
    
    # Initialize result data
    result_data = {
        'medical_summary': 'No medical summary available.',
        'sources': [],
        'tool_info': {}
    }
    
    # Check if this is an enhanced HTML response
    if '<div' in answer and '<h4' in answer:
        # Parse HTML content to extract sections
        import re
        
        # Extract Medical Summary
        medical_summary_match = re.search(r'<h4[^>]*>.*?Medical Summary.*?</h4><div[^>]*>(.*?)</div>', answer, re.DOTALL | re.IGNORECASE)
        if medical_summary_match:
            medical_content = medical_summary_match.group(1)
            # Clean HTML tags but preserve content
            medical_content = re.sub(r'<[^>]+>', '', medical_content).strip()
            result_data['medical_summary'] = medical_content
        
        # Extract Sources
        sources_match = re.search(r'<h4[^>]*>.*?Sources.*?</h4><div[^>]*>(.*?)</div>', answer, re.DOTALL | re.IGNORECASE)
        if sources_match:
            sources_content = sources_match.group(1)
            # Extract links and convert to simple format
            link_matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>[^<]*\(([^)]+)\)', sources_content)
            for url, title, source_type in link_matches:
                result_data['sources'].append(f"{title} ({source_type})")
            
            # If no links found, extract plain text
            if not result_data['sources']:
                sources_text = re.sub(r'<[^>]+>', '', sources_content).strip()
                if sources_text:
                    result_data['sources'].append(sources_text)
        
        # Extract Tool Selection info from HTML
        tool_match = re.search(r'<h4[^>]*>.*?Tool Selection.*?</h4><div[^>]*>(.*?)</div>', answer, re.DOTALL | re.IGNORECASE)
        if tool_match:
            tool_content = tool_match.group(1)
            
            # Extract Primary Tool
            primary_tool_match = re.search(r'Primary Tool:\s*<strong>([^<]+)</strong>', tool_content)
            primary_tool = primary_tool_match.group(1) if primary_tool_match else routing_info.get('primary_tool', 'Unknown')
            
            # Extract Confidence
            confidence_match = re.search(r'Confidence:\s*<span[^>]*>([^<]+)</span>', tool_content)
            confidence = confidence_match.group(1) if confidence_match else routing_info.get('confidence', 'Unknown')
            
            # Extract Reasoning
            reasoning_match = re.search(r'Reasoning:\s*([^<]+?)(?:<|$)', tool_content)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else routing_info.get('reasoning', 'No reasoning provided.')
            
            result_data['tool_info'] = {
                'primary_tool': primary_tool,
                'confidence': confidence,
                'reasoning': reasoning
            }
    else:
        # Handle plain text responses
        result_data['medical_summary'] = answer[:500] + "..." if len(answer) > 500 else answer
        
        # Add basic tool info from routing_info
        primary_tool = routing_info.get('primary_tool', 'Unknown')
        confidence = routing_info.get('confidence', 'Unknown')
        
        # Format confidence
        if isinstance(confidence, str):
            confidence_mapping = {'high': 'High (≈90%)', 'medium': 'Medium (≈70%)', 'low': 'Low (≈50%)'}
            confidence = confidence_mapping.get(confidence.lower(), confidence)
        elif isinstance(confidence, (int, float)):
            confidence = f"{confidence}%"
        
        result_data['tool_info'] = {
            'primary_tool': primary_tool,
            'confidence': confidence,
            'reasoning': routing_info.get('reasoning', explanation or 'No reasoning provided.')
        }
        
        result_data['sources'] = ['No specific sources identified.']
    
    return result_data

@app.route("/data", methods=["POST"])
def handle_query():
    """Original JSON endpoint for the UI - returns JSON responses"""
    user_input = request.json.get("data", "")
    patient_problem = request.json.get("patient_problem", "").strip()
    
    if not user_input:
        return jsonify({
            "response": False,
            "message": "Please provide a valid input to get a medical response."
        })
    
    # Store patient context for system message (don't append to query)
    if patient_problem:
        print(f"📋 Using patient context as system message: '{patient_problem}'")
    
    # Use original user input (not contextual_input)
    query_input = user_input

    try:
        # 🎯 INTEGRATED MEDICAL RAG SYSTEM WITH INTELLIGENT TOOL ROUTING
        if integrated_rag_system and INTEGRATED_RAG_AVAILABLE:
            print("🚀 Using Integrated Medical RAG System with Tool Routing")
            print(f"📝 Query: '{user_input}'")
            if patient_problem:
                print(f"📋 Patient Context: '{patient_problem}'")
            
            # Extract session ID from request if available, fallback to current session
            session_id = request.json.get("session_id")
            if not session_id or session_id == "guest":
                # Use the current session folder created when page was loaded
                global last_created_folder
                session_id = last_created_folder if last_created_folder else "guest"
                print(f"🔄 Using current session: {session_id}")
            
            # Use the integrated RAG system's intelligent query method
            integrated_result = integrated_rag_system.query(query_input, session_id, patient_problem)
            
            if integrated_result and integrated_result.get('answer'):
                answer = integrated_result['answer']
                routing_info = integrated_result.get('routing_info', {})
                tools_used = integrated_result.get('tools_used', [])
                explanation = integrated_result.get('explanation', '')
                
                # Return JSON response for UI
                return jsonify({
                    "response": True,
                    "message": answer,
                    "routing_details": {
                        "disciplines": tools_used,
                        "sources": routing_info.get('sources', []),
                        "method": routing_info.get('confidence', 'medium'),
                        "confidence": routing_info.get('confidence', 'medium')
                    }
                })
            else:
                print("⚠️ No response from Integrated RAG system, falling back to Two-Store RAG")
        
        # 🚀 FALLBACK: TWO-STORE RAG ARCHITECTURE WITH LEXICAL GATE
        if rag_manager and RAG_ARCHITECTURE_AVAILABLE:
            print("🧠 Using Two-Store RAG Architecture with Lexical Gate")
            print(f"📝 Query: '{user_input}'")
            if patient_problem:
                print(f"📋 Patient Context: '{patient_problem}'")
            
            # Extract session ID from request if available
            session_id = request.json.get("session_id", "guest")
            
            # Use the RAG manager's intelligent routing with session-specific vector DB
            rag_result = rag_manager.query_with_routing(query_input, session_id)
            
            if rag_result['responses']:
                # Sort responses by confidence
                rag_result['responses'].sort(key=lambda x: x["confidence"], reverse=True)
                
                # Create comprehensive response
                final_response = ""
                
                for i, resp in enumerate(rag_result['responses'][:2], 1):  # Limit to top 2 responses
                    if i > 1:
                        final_response += "\n\n"  # Add spacing between multiple responses
                    final_response += clean_response_text(resp['content'])
                
                # Add citations in bold
                if rag_result['citations']:
                    final_response += "\n\n**Citations:**\n"
                    for citation in rag_result['citations']:
                        final_response += f"{citation}\n"
                
                # Add routing information
                routing_info = rag_result['routing_info']
                sources_info = ', '.join(routing_info.get('sources_queried', []))
                final_response += f"\n**RAG Routing:** TF-IDF similarity: {routing_info.get('similarity_score', 0):.3f}, Sources: {sources_info}"
                
                # Return JSON response for two-store RAG
                return jsonify({
                    "response": True,
                    "message": final_response,
                    "routing_details": {
                        "disciplines": routing_info.get('sources_queried', []),
                        "sources": rag_result['citations'],
                        "method": "Two-Store RAG",
                        "confidence": f"{routing_info.get('similarity_score', 0):.0%}"
                    }
                })
            else:
                # No responses from RAG system, fallback to original implementation
                print("⚠️ No responses from RAG system, falling back to legacy implementation")
        
        # 🏥 LEGACY IMPLEMENTATION: Keep existing medical routing as fallback
        print("🔄 Using legacy medical routing system")
        
        # INTELLIGENT ROUTING: Analyze query to determine relevant disciplines
        routing_result = medical_router.analyze_query(query_input)
        relevant_disciplines = routing_result["disciplines"]
        confidence_scores = routing_result["confidence_scores"]
        
        print(f"🧠 Query: '{user_input}'")
        if patient_problem:
            print(f"📋 Patient Context: '{patient_problem}'")
        print(f"🎯 Routed to disciplines: {relevant_disciplines}")
        print(f"📊 Confidence scores: {confidence_scores}")
        
        # Collect responses from multiple sources
        all_responses = []
        all_citations = []
        
        # 1. Query Organization KB (discipline-specific) - Skip session-based disciplines
        for discipline_id in relevant_disciplines:
            # Skip session-based disciplines for Organization KB
            discipline_config = next((d for d in disciplines_config.get("disciplines", []) if d["id"] == discipline_id), None)
            if discipline_config and discipline_config.get("is_session_based", False):
                continue
                
            try:
                vector_db_path = get_discipline_vector_db_path(discipline_id)
                if vector_db_path and os.path.exists(vector_db_path):
                    print(f"🏥 Querying Organization KB: {discipline_id}")
                    
                    vector_store = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
                    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                    
                    # Create contextual LLM with patient context as system message
                    contextual_llm = create_contextual_llm(patient_problem)
                    contextual_qa_chain = RetrievalQA.from_chain_type(llm=contextual_llm, retriever=retriever)
                    
                    org_response = contextual_qa_chain.invoke(query_input)
                    search_results = retriever.invoke(query_input)
                    
                    if org_response['result'].strip():
                        all_responses.append({
                            "source": f"Organization KB - {discipline_id.replace('_', ' ').title()}",
                            "content": org_response['result'],
                            "confidence": confidence_scores.get(discipline_id, 70)
                        })
                        
                        org_citations = enhance_with_citations(search_results)
                        if org_citations != "No citations available":
                            clean_citations = clean_response_text(org_citations)
                            # Split individual citations and format each one
                            citation_lines = [line.strip() for line in clean_citations.split('\n') if line.strip()]
                            for citation_line in citation_lines:
                                all_citations.append(f"**{discipline_id.replace('_', ' ').title()}: {citation_line}**")
                            
            except Exception as e:
                print(f"Error querying {discipline_id}: {e}")
        
        # 2. Query Adhoc KB (user-uploaded content) - Higher priority if doctors_files is selected
        doctors_files_selected = "doctors_files" in relevant_disciplines
        try:
            latest_vector_db = get_latest_vector_db()
            if latest_vector_db and os.path.exists(latest_vector_db):
                if doctors_files_selected:
                    print("📄 Querying Doctor's Files (prioritized)")
                else:
                    print("📄 Querying Adhoc KB (user uploads)")
                
                vector_store = Chroma(persist_directory=latest_vector_db, embedding_function=embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                
                # Create contextual LLM with patient context as system message  
                contextual_llm = create_contextual_llm(patient_problem)
                contextual_qa_chain = RetrievalQA.from_chain_type(llm=contextual_llm, retriever=retriever)
                
                adhoc_response = contextual_qa_chain.invoke(query_input)
                search_results = retriever.invoke(query_input)
                
                if adhoc_response['result'].strip():
                    source_name = "Doctor's Files" if doctors_files_selected else "User Uploaded Documents"
                    confidence = confidence_scores.get("doctors_files", 85) if doctors_files_selected else 85
                    
                    all_responses.append({
                        "source": source_name,
                        "content": adhoc_response['result'],
                        "confidence": confidence
                    })
                    
                    adhoc_citations = enhance_with_citations(search_results)
                    if adhoc_citations != "No citations available":
                        clean_citations = clean_response_text(adhoc_citations)
                        # Split individual citations and format each one
                        citation_lines = [line.strip() for line in clean_citations.split('\n') if line.strip()]
                        citation_prefix = "Doctor's Files" if doctors_files_selected else "User Documents"
                        for citation_line in citation_lines:
                            all_citations.append(f"**{citation_prefix}: {citation_line}**")
                        
        except Exception as e:
            print(f"Error querying adhoc KB: {e}")
        
        # 3. Synthesize final response
        if all_responses:
            # Sort responses by confidence
            all_responses.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Create comprehensive response - just the content without headers
            final_response = ""
            
            for i, resp in enumerate(all_responses[:2], 1):  # Limit to top 2 responses
                if i > 1:
                    final_response += "\n\n"  # Add spacing between multiple responses
                final_response += clean_response_text(resp['content'])
            
            # Add citations in bold
            if all_citations:
                final_response += "\n\n**Citations:**\n"
                for citation in all_citations:
                    final_response += f"{citation}\n"
            
            # Add routing information on a new line
            routing_info = f"\n**Query Routing:** Analyzed and routed to {', '.join([d.replace('_', ' ').title() for d in relevant_disciplines])}"
            final_response += routing_info
            
            # Return response with routing details
            # Return JSON response for legacy medical routing
            return jsonify({
                "response": True,
                "message": final_response,
                "routing_details": {
                    "disciplines": relevant_disciplines,
                    "sources": all_citations,
                    "method": routing_result.get('routing_method', 'hybrid'),
                    "confidence": f"{max(confidence_scores.values()) if confidence_scores else 0:.0%}"
                }
            })
        else:
            # Fallback to general response
            fallback_response = f"""
            I understand you're asking about: "{user_input}"
            
            However, I couldn't find specific information in the available medical knowledge bases for the disciplines I identified: {', '.join([d.replace('_', ' ').title() for d in relevant_disciplines])}.
            
            This could be because:
            1. The Organization KB doesn't have information on this specific topic yet
            2. No user documents have been uploaded that relate to this query
            3. The query might need to be more specific
            
            **Query was routed to:** {', '.join([d.replace('_', ' ').title() for d in relevant_disciplines])}
            
            Consider uploading relevant medical documents or rephrasing your question for better results.
            """
            
            # Return JSON response for fallback
            return jsonify({
                "response": True,
                "message": clean_response_text(fallback_response),
                "routing_details": {
                    "disciplines": relevant_disciplines,
                    "sources": [],
                    "method": routing_result.get('routing_method', 'hybrid'),
                    "confidence": "Low (≈30%)"
                }
            })

    except Exception as e:
        print(f"Error in handle_query: {e}")
        # Return JSON error response
        return jsonify({
            "response": False,
            "message": f"An error occurred while processing your query: {str(e)}"
        })


@app.route("/data-html", methods=["POST"])
def handle_query_html():
    """HTML endpoint that returns complete HTML documents with 3-section structure"""
    user_input = request.json.get("data", "")
    patient_problem = request.json.get("patient_problem", "").strip()
    
    if not user_input:
        # Create HTML error response for empty input
        result_data = {
            'medical_summary': 'Please provide a valid input to get a medical response.',
            'sources': ['Input Validation'],
            'tool_info': {
                'primary_tool': 'Input Validator',
                'confidence': 'N/A',
                'reasoning': 'No query text provided in the request.'
            }
        }
        
        html_response = generate_full_html_response(result_data)
        from flask import make_response
        response = make_response(html_response)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        return response

    # Store patient context for system message (don't append to query)
    if patient_problem:
        print(f"📋 Using patient context as system message: '{patient_problem}'")
    
    # Use original user input (not contextual_input)
    query_input = user_input

    try:
        # 🎯 INTEGRATED MEDICAL RAG SYSTEM WITH INTELLIGENT TOOL ROUTING
        if integrated_rag_system and INTEGRATED_RAG_AVAILABLE:
            print("🚀 Using Integrated Medical RAG System with Tool Routing")
            print(f"📝 Query: '{user_input}'")
            if patient_problem:
                print(f"📋 Patient Context: '{patient_problem}'")
            
            # Extract session ID from request if available, fallback to current session
            session_id = request.json.get("session_id")
            if not session_id or session_id == "guest":
                # Use the current session folder created when page was loaded
                global last_created_folder
                session_id = last_created_folder if last_created_folder else "guest"
                print(f"🔄 Using current session: {session_id}")
            
            # Use the integrated RAG system's intelligent query method
            answer, routing_info, tools_used, explanation = integrated_rag_system.intelligent_query(
                query_input, 
                session_id=session_id
            )
            
            # Handle Enhanced Tools Response (Wikipedia/ArXiv with HTML formatting)
            if tools_used and any('Enhanced' in tool for tool in tools_used):
                print("📋 Processing Enhanced Tools Response (HTML format)")
                
                # Parse the enhanced HTML response using our new parser
                result_data = parse_enhanced_response(answer, routing_info, tools_used, explanation)
                
                html_response = generate_full_html_response(result_data)
                from flask import make_response
                response = make_response(html_response)
                response.headers['Content-Type'] = 'text/html; charset=utf-8'
                return response
            
            # Handle Two-Store RAG Response
            else:
                print("📋 Processing Two-Store RAG Response")
                
                # For internal VectorDB or organization KB responses, create result_data
                result_data = {
                    'Answer': answer,
                    'sources': ['Internal Document (PDF)', 'Organization Knowledge Base'] if 'organization' in tools_used[0].lower() else ['Internal Document (PDF)'],
                    'tool_info': {
                        'primary_tool': tools_used[0] if tools_used else 'Internal_VectorDB',
                        'confidence': f"{routing_info.get('confidence', 'medium').title()} (≈70%)",
                        'reasoning': explanation or f"Queried internal knowledge base due to uploaded content. {routing_info.get('reasoning', '')}"
                    }
                }
                
                html_response = generate_full_html_response(result_data)
                from flask import make_response
                response = make_response(html_response)
                response.headers['Content-Type'] = 'text/html; charset=utf-8'
                return response

        # 🔄 FALLBACK: Return error if integrated system not available
        else:
            print("⚠️ Integrated RAG system not available for HTML endpoint")
            
            # Create error response
            result_data = {
                'medical_summary': 'The HTML endpoint requires the integrated RAG system to be available. Please use the regular /data endpoint.',
                'sources': ['System Configuration'],
                'tool_info': {
                    'primary_tool': 'Error Handler',
                    'confidence': 'N/A',
                    'reasoning': 'HTML endpoint requires integrated RAG system which is not currently available.'
                }
            }
            
            html_response = generate_full_html_response(result_data)
            from flask import make_response
            response = make_response(html_response)
            response.headers['Content-Type'] = 'text/html; charset=utf-8'
            return response

    except Exception as e:
        print(f"❌ Error in HTML query handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create HTML error response
        result_data = {
            'medical_summary': f'An error occurred while processing your query: {str(e)}',
            'sources': ['System Error'],
            'tool_info': {
                'primary_tool': 'Error Handler',
                'confidence': 'N/A',
                'reasoning': 'An unexpected error occurred during query processing.'
            }
        }
        
        html_response = generate_full_html_response(result_data)
        from flask import make_response
        response = make_response(html_response)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        return response


def initialize_session(user="guest"):
    """Initializes a new session folder when the page is refreshed."""
    global last_created_folder
    timestamp = get_timestamp()
    last_created_folder = f"{user}_{timestamp}"  # Format: user_MMDDYYYYHHMM

    # Ensure required directories exist
    os.makedirs(os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder), exist_ok=True)
    os.makedirs(os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder), exist_ok=True)

    print(f"📂 New session folder created: {last_created_folder}")
    return last_created_folder


@app.route('/', methods=['GET'])
def index():
    """Refresh page to create a new session folder."""
    user = request.args.get('user', 'guest')
    initialize_session(user)
    return render_template("index.html")

@app.route('/api/disciplines', methods=['GET'])
def get_disciplines():
    """Return available disciplines for UI dropdown."""
    try:
        disciplines = get_available_disciplines()
        return jsonify({
            "success": True,
            "disciplines": disciplines,
            "selection_rules": disciplines_config.get("selection_rules", {})
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/validate_disciplines', methods=['POST'])
def validate_disciplines():
    """Validate selected disciplines."""
    try:
        selected = request.json.get("selected_disciplines", [])
        is_valid, message = validate_discipline_selection(selected)
        return jsonify({
            "success": True,
            "is_valid": is_valid,
            "message": message
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def count_files_in_folder(folder):
    """Returns the number of files in a given folder."""
    if os.path.exists(folder):
        return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    return 0

def can_upload_more_files(new_files_count):
    """Check if the total PDFs and URLs in the session folder exceed the limit (10)."""
    pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder)
    url_folder = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)
    
    pdf_count = count_files_in_folder(pdf_folder)
    url_count = count_files_in_folder(url_folder)
    
    return (pdf_count + url_count + new_files_count) <= 10  # Limit is 10 files total

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Upload multiple PDF files to the session folder."""
    global last_created_folder
    if not last_created_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    files = request.files.getlist('file')  # Get multiple files
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No valid files uploaded"}), 400

    if not can_upload_more_files(len(files)):
        return jsonify({"message": "Cannot process: Maximum of 10 PDFs exceeded."})

    pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder)
    os.makedirs(pdf_folder, exist_ok=True)

    saved_files = []
    
    try:
        for file in files:
            file_path = os.path.join(pdf_folder, file.filename)
            file.save(file_path)
            saved_files.append(file.filename)

        return jsonify({"message": "PDFs uploaded successfully", "files": saved_files})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/upload_url', methods=['POST'])
def upload_url():
    """Upload a text file containing URLs."""
    global last_created_folder
    if not last_created_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400
    


    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty file uploaded"}), 400

        url_folder = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)
        os.makedirs(url_folder, exist_ok=True)
        file_path = os.path.join(url_folder, "urls.txt")
        file.save(file_path)

        # Read URLs
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f.readlines() if line.strip()]

        if not urls:
            return jsonify({"error": "Uploaded file contains no valid URLs"}), 400
        if len(urls) > 3:
            return jsonify({"message": "Cannot process: Maximum of 3 URLs exceeded."})

        return jsonify({"message": "URLs uploaded successfully", "file": file_path, "url_count": len(urls)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/upload_organization_kb', methods=['POST'])
def upload_organization_kb():
    """Upload documents to Organization KB for specific disciplines."""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        discipline_id = request.form.get('discipline_id')
        if not discipline_id:
            return jsonify({"error": "No discipline specified"}), 400
        
        # Validate discipline
        discipline_path = None
        for discipline in disciplines_config.get("disciplines", []):
            if discipline["id"] == discipline_id:
                discipline_path = discipline.get("kb_path")
                break
        
        if not discipline_path:
            return jsonify({"error": f"Invalid discipline: {discipline_id}"}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No valid files uploaded"}), 400
        
        # Create discipline directory
        discipline_dir = os.path.join(".", discipline_path)
        os.makedirs(discipline_dir, exist_ok=True)
        
        saved_files = []
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=128)
        
        for file in files:
            if file.filename.endswith('.pdf'):
                file_path = os.path.join(discipline_dir, file.filename)
                file.save(file_path)
                saved_files.append(file.filename)
                
                # Extract text and create documents
                text_content = extract_text_from_pdf(file_path)
                if text_content:
                    if isinstance(text_content, str):
                        text_content = [text_content]
                    
                    for page_num, text in enumerate(text_content, start=1):
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            documents.append(Document(
                                page_content=chunk,
                                metadata={
                                    "source": file.filename,
                                    "type": "organization_pdf",
                                    "discipline": discipline_id,
                                    "page": page_num
                                }
                            ))
        
        # Create/update vector database for this discipline
        if documents:
            create_organization_vector_db(discipline_id, documents)
            
        return jsonify({
            "message": f"Successfully uploaded {len(saved_files)} files to {discipline_id}",
            "files": saved_files,
            "documents_created": len(documents)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/create_vector_db', methods=['POST'])
def create_vector_db():
    """Parse PDFs and URLs, then create a Chroma Vector Database."""
    global last_created_folder
    if not last_created_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400

    try:
        persist_dir = os.path.join(VECTOR_DB_PATH, last_created_folder)
        os.makedirs(persist_dir, exist_ok=True)

        pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder)
        url_folder = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)

        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')] if os.path.exists(pdf_folder) else []
        url_files = [os.path.join(url_folder, f) for f in os.listdir(url_folder)] if os.path.exists(url_folder) else []

        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=128)

        for pdf_file in pdf_files:
                text_content = extract_text_from_pdf(pdf_file)
                if not text_content:
                    print(f"No text extracted from {pdf_file}")
                    continue

                if isinstance(text_content, str):  # Convert single string to a list
                    text_content = [text_content]
                print(text_content)
                for text in text_content:
                    chunks = text_splitter.split_text(text)
                    for page_num, chunk in enumerate(chunks, start=1):  # `enumerate` starts from 1
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": pdf_file,  # File path
                                "type": "pdf",        # Ensure type is set
                                "page": page_num      # Correct page number
                            }
                        ))


        #  Process URLs
        for url_file in url_files:
            with open(url_file, 'r', encoding='utf-8') as file:
                urls = [line.strip() for line in file.readlines() if line.strip()]

            for url in urls:
                try:
                    text = extract_text_from_url(url)
                    if not text:
                        print(f" No text extracted from {url}")
                        continue

                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": url,
                                "type": "url"
                            }
                        ))


                except Exception as e:
                    print(f"Error extracting text from {url}: {str(e)}")

        if not documents:
            return jsonify({"error": "No valid documents found to create the database"}), 400
        
        print(f"{len(documents)} documents prepared for vectorization")

        # Convert to Chroma format
        vector_store = Chroma.from_documents(
            documents,  # Use the list of Document objects directly
            embedding=embeddings,
            persist_directory=persist_dir
        )

        # 🚀 NEW: Add documents to RAG manager's local knowledge base
        if rag_manager and RAG_ARCHITECTURE_AVAILABLE:
            try:
                print("📚 Adding documents to RAG manager's local knowledge base...")
                rag_manager.add_documents_to_local(documents)
                print(f"✅ Successfully added {len(documents)} documents to kb_local and updated lexical gate")
            except Exception as e:
                print(f"⚠️ Error adding documents to RAG manager: {e}")

        return jsonify({"message": "New Vector DB created", "db": last_created_folder})

    except Exception as e:
        print(f" Error in /create_vector_db: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/generate_patient_pdf", methods=["POST"])
def generate_patient_pdf():
    """Generate PDF for patient notes with patient problem capture and Azure upload"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.colors import black
        import io
        import os
        
        data = request.json
        
        # Extract data
        doctor_name = data.get('doctorName', '')
        patient_name = data.get('patientName', '')
        patient_id = data.get('patientId', '')
        date_time = data.get('dateTime', '')
        transcription = data.get('transcription', '')
        summary = data.get('summary', '')
        conclusion = data.get('conclusion', '')
        
        # NEW: Extract patient problem (required for first PDF only)
        patient_problem = data.get('patientProblem', '')
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch, leftMargin=inch, 
                               topMargin=inch, bottomMargin=inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=10,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            alignment=TA_CENTER,
            fontName='Helvetica',
            textColor='blue'
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        session_style = ParagraphStyle(
            'SessionStyle',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            spaceAfter=5
        )
        
        # Build PDF content
        story = []
        
        # Add logo at the top right with patient-specific title
        logo_path = os.path.join(os.path.dirname(__file__), 'ClientLogo101.png')
        patient_display_name = patient_name if patient_name else "Patient Name"
        main_title = f"Patient – {patient_display_name} – Recording Notes"
        
        if os.path.exists(logo_path):
            try:
                # Create a table to position title on left and logo on right
                logo_img = Image(logo_path, width=1.5*inch, height=0.9*inch)
                title_paragraph = Paragraph(main_title, title_style)
                
                # Create table with title and logo
                header_data = [[title_paragraph, logo_img]]
                header_table = Table(header_data, colWidths=[4.5*inch, 2*inch])
                header_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                story.append(header_table)
                story.append(Spacer(1, 15))
            except Exception as e:
                print(f"Warning: Could not add logo to PDF: {e}")
                # Fallback to just title
                story.append(Paragraph(main_title, title_style))
                story.append(Spacer(1, 15))
        else:
            # Fallback to just title if logo doesn't exist
            story.append(Paragraph(main_title, title_style))
            story.append(Spacer(1, 15))
        
        # Add "Patient Recording Notes" subtitle
        story.append(Paragraph("Patient Recording Notes", subtitle_style))
        story.append(Spacer(1, 10))
        
        # Patient and Doctor Information
        patient_id_display = patient_id if patient_id else "N/A"
        patient_info = f"Patient Name: {patient_display_name} – Patient ID: {patient_id_display}"
        doctor_info = f"Doctor's Name: {doctor_name if doctor_name else 'Unknown Doctor'}"
        
        story.append(Paragraph(patient_info, styles['Normal']))
        story.append(Spacer(1, 5))
        story.append(Paragraph(doctor_info, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Session Information
        story.append(Paragraph("Session Information", header_style))
        story.append(Paragraph("Transcription Engine: Whisper; Summary Engine: OpenAI", session_style))
        
        # Format date
        display_date = date_time if date_time else datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        story.append(Paragraph(f"Date: {display_date}", session_style))
        story.append(Spacer(1, 15))
        
        # NEW: Patient Problem (if provided)
        if patient_problem:
            story.append(Paragraph("Patient Problem", header_style))
            story.append(Paragraph(patient_problem, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Original Transcription Text
        story.append(Paragraph("Original Transcription Text", header_style))
        transcription_text = transcription if transcription else "No transcription available"
        story.append(Paragraph(transcription_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Medical Summary
        story.append(Paragraph("Medical Summary", header_style))
        summary_text = summary if summary else "No summary available"
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Conclusion & Recommendations
        story.append(Paragraph("Conclusion & Recommendations", header_style))
        conclusion_text = conclusion if conclusion else "No conclusion available"
        story.append(Paragraph(conclusion_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{patient_name.upper().replace(' ', '')}-{timestamp}.pdf"
        
        # NEW: Upload to Azure if available
        azure_url = None
        if AZURE_AVAILABLE:
            try:
                # Prepare metadata
                metadata = {
                    'doctor_name': doctor_name,
                    'patient_name': patient_name,
                    'patient_id': patient_id,
                    'date_time': date_time,
                    'patient_problem': patient_problem,
                    'pdf_type': 'patient_notes',
                    'generated_at': datetime.now().isoformat()
                }
                
                # Upload to patient summary container
                azure_url = upload_pdf_to_azure(pdf_content, filename, "patient_summary", metadata)
                
            except Exception as e:
                print(f"Azure upload failed: {e}")
        
        # Return PDF as response
        from flask import make_response
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        # Add Azure URL to response headers if available
        if azure_url:
            response.headers['X-Azure-URL'] = azure_url
        
        return response
        
    except ImportError:
        # Fallback if reportlab is not installed
        return jsonify({"error": "PDF generation not available. Please install reportlab."}), 500
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500


@app.route("/generate_chat_pdf", methods=["POST"])
def generate_chat_pdf():
    """Generate PDF for chat conversation with specified format"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.colors import black
        import io
        import os
        
        data = request.json

        # Extract data
        doctor_name = data.get('doctorName', 'Dr. Name')
        patient_name = data.get('patientName', '')
        patient_id = data.get('patientId', '')
        patient_problem = data.get('patientProblem', '')
        messages = data.get('messages', [])
        json_data = data.get('jsonData', '')
        
        # Debug logging
        print(f"DEBUG: Received data in generate_chat_pdf:")
        print(f"  doctor_name: {doctor_name}")
        print(f"  patient_name: {patient_name}")
        print(f"  patient_id: {patient_id}")
        print(f"  patient_problem: '{patient_problem}'")
        print(f"  messages count: {len(messages)}")
        print(f"  json_data: {json_data}")
        print(f"DEBUG: Full request data: {data}")  # Show full JSON

        # Generate timestamp
        now = datetime.now()
        formatted_date = now.strftime("%m/%d/%Y, %I:%M:%S %p")

        # Create PDF filename - include patient name if provided
        timestamp = now.strftime("%Y%m%d%H%M")
        if patient_name:
            filename = f"{doctor_name.upper().replace(' ', '')}-{patient_name.upper().replace(' ', '')}-{timestamp}.pdf"
        else:
            filename = f"{doctor_name.upper().replace(' ', '')}-{timestamp}.pdf"

        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch, leftMargin=inch,
                               topMargin=inch, bottomMargin=inch)

        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=10,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            alignment=TA_CENTER,
            fontName='Helvetica',
            textColor='blue'
        )

        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )

        session_style = ParagraphStyle(
            'SessionStyle',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            spaceAfter=5
        )

        content_style = ParagraphStyle(
            'ContentStyle',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            spaceAfter=8,
            alignment=TA_LEFT,
            spaceBefore=2,
            leading=14  # Increase line spacing for better readability
        )

        # Style for citation content with better formatting
        citation_style = ParagraphStyle(
            'CitationStyle',
            parent=styles['Normal'],
            fontSize=9,
            fontName='Helvetica',
            spaceAfter=4,
            alignment=TA_LEFT,
            leftIndent=10,
            leading=12
        )

        section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            spaceAfter=6,
            spaceBefore=12,
            alignment=TA_CENTER
        )

        # Build PDF content
        story = []

        # Title with optional patient name
        patient_display_name = patient_name if patient_name else "Patient Name"
        main_title = f"Patient – {patient_display_name} – Research"

        logo_path = os.path.join(os.path.dirname(__file__), 'ClientLogo101.png')
        # Create a table to position title on left and logo on right (match patient PDF layout)
        if os.path.exists(logo_path):
            try:
                logo_img = Image(logo_path, width=1.5*inch, height=0.9*inch)
                title_paragraph = Paragraph(main_title, title_style)

                header_data = [[title_paragraph, logo_img]]
                header_table = Table(header_data, colWidths=[4.5*inch, 2*inch])
                header_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                story.append(header_table)
                story.append(Spacer(1, 15))
            except Exception as e:
                print(f"Warning: Could not add logo to PDF: {e}")
                story.append(Paragraph(main_title, title_style))
                story.append(Spacer(1, 15))
        else:
            story.append(Paragraph(main_title, title_style))
            story.append(Spacer(1, 15))

        # Subtitle
        story.append(Paragraph("Patient Recording Notes", subtitle_style))
        story.append(Spacer(1, 10))

        # Patient and Doctor Information
        # Only include patient ID if provided (avoid showing 'N/A')
        if patient_id:
            patient_info = f"Patient Name: {patient_display_name} – Patient ID: {patient_id}"
        else:
            patient_info = f"Patient Name: {patient_display_name}"
        doctor_info = f"Doctor's Name: {doctor_name if doctor_name else 'Unknown Doctor'}"

        story.append(Paragraph(patient_info, styles['Normal']))
        story.append(Spacer(1, 5))
        story.append(Paragraph(doctor_info, styles['Normal']))
        story.append(Spacer(1, 15))

        # Session Information
        story.append(Paragraph("Session Information", header_style))
        story.append(Paragraph(f"Date: {formatted_date}", session_style))
        story.append(Spacer(1, 12))

        # Patient Problem (if provided)
        if patient_problem:
            print(f"DEBUG: Adding Patient Problem section: '{patient_problem}'")
            story.append(Paragraph("Patient Problem", header_style))
            story.append(Paragraph(patient_problem, styles['Normal']))
            story.append(Spacer(1, 12))
        else:
            print("DEBUG: No patient problem provided, skipping section")

        # JSON Data section (if provided)
        if json_data:
            story.append(Paragraph("JSON Data:", section_header_style))
            story.append(Paragraph(json_data, content_style))
            story.append(Spacer(1, 12))

        # Conversation
        story.append(Paragraph("Conversation", header_style))
        story.append(Spacer(1, 6))

        def convert_markdown_to_reportlab(text):
            """Convert enhanced tools HTML format to ReportLab-compatible text"""
            import re
            from bs4 import BeautifulSoup
            
            # Check if this is HTML content (contains div tags)
            if '<div' in text and '<h4' in text:
                try:
                    # Parse HTML using BeautifulSoup
                    soup = BeautifulSoup(text, 'html.parser')
                    
                    sections = {
                        'answer': '',
                        'source': '',
                        'tool_routing': ''
                    }
                    
                    # Extract sections by looking for h4 headers
                    divs = soup.find_all('div', style=lambda x: x and 'margin-bottom' in x)
                    
                    for div in divs:
                        h4 = div.find('h4')
                        if h4:
                            header_text = h4.get_text().strip().lower()
                            content_div = div.find('div', style=lambda x: x and ('background-color' in x or 'padding' in x))
                            
                            if 'answer' in header_text:
                                if content_div:
                                    sections['answer'] = content_div.get_text().strip()
                            elif 'source' in header_text:
                                if content_div:
                                    # Extract links and text
                                    sources = []
                                    links = content_div.find_all('a')
                                    if links:
                                        for link in links:
                                            link_text = link.get_text().strip()
                                            # Get the text after the link (like "(Wikipedia)")
                                            next_text = link.next_sibling
                                            if next_text and isinstance(next_text, str):
                                                sources.append(f"{link_text} {next_text.strip()}")
                                            else:
                                                sources.append(link_text)
                                    else:
                                        # Fallback to plain text
                                        sources = [content_div.get_text().strip()]
                                    sections['source'] = '\n'.join(sources)
                        elif 'tool selection' in header_text or 'routing' in header_text:
                            if content_div:
                                # Get the raw text and preserve structure better
                                routing_html = str(content_div)
                                sections['tool_routing'] = routing_html                    # Build formatted text
                    formatted_parts = []
                    
                    # 1. Answer section (main content)
                    if sections['answer']:
                        formatted_parts.append(sections['answer'])
                    
                    # 2. Source section
                    if sections['source']:
                        sources = [s.strip() for s in sections['source'].split('\n') if s.strip()]
                        if sources:
                            formatted_parts.append(f"<br/><br/><b>Sources:</b><br/>• " + "<br/>• ".join(sources))
                    
                    # 3. Tool routing section
                    if sections['tool_routing']:
                        routing_html = sections['tool_routing']
                        
                        # Simple and reliable parsing approach
                        routing_soup = BeautifulSoup(routing_html, 'html.parser')
                        text = routing_soup.get_text()
                        
                        confidence = ""
                        tools_used = ""
                        reasoning = ""
                        
                        # Split by key sections and parse each part
                        lines = text.replace('\n', ' ').split('Tools Used:')
                        
                        if len(lines) >= 2:
                            # First part contains confidence
                            confidence_part = lines[0]
                            confidence_match = re.search(r'Confidence:\s*(.+?)$', confidence_part.strip(), re.IGNORECASE)
                            if confidence_match:
                                confidence = confidence_match.group(1).strip()
                            
                            # Second part contains tools used and reasoning
                            rest = lines[1]
                            reasoning_split = rest.split('Reasoning:')
                            
                            if len(reasoning_split) >= 2:
                                tools_used = reasoning_split[0].strip()
                                reasoning = reasoning_split[1].strip()
                            else:
                                tools_used = rest.strip()
                        
                        # Build routing section
                        routing_parts = []
                        if confidence:
                            routing_parts.append(f"<b>Confidence:</b> {confidence}")
                        if tools_used:
                            routing_parts.append(f"<b>Tools Used:</b> {tools_used}")
                        if reasoning:
                            routing_parts.append(f"<b>Reasoning:</b> {reasoning}")
                        
                        if routing_parts:
                            formatted_parts.append(f"<br/><br/><b>Tool Selection & Query Routing:</b><br/>" + "<br/>".join(routing_parts))
                    
                    # Combine all parts
                    full_text = "".join(formatted_parts)
                    
                    # Clean up extra spaces
                    full_text = re.sub(r'\s+', ' ', full_text)
                    
                    return full_text.strip()
                    
                except Exception as e:
                    print(f"Error parsing HTML content: {e}")
                    # Fallback to plain text extraction
                    soup = BeautifulSoup(text, 'html.parser')
                    return soup.get_text().strip()
            
            else:
                # Handle plain text format (original logic)
                sections = {
                    'answer': '',
                    'source': '',
                    'tool_routing': ''
                }
                
                # Split text by the new section headers
                lines = text.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check for section headers
                    if line.lower().startswith('answer'):
                        current_section = 'answer'
                        continue
                    elif line.lower().startswith('source'):
                        current_section = 'source'
                        continue
                    elif line.lower().startswith('tool selection') or line.lower().startswith('confidence:'):
                        current_section = 'tool_routing'
                        if line.lower().startswith('tool selection'):
                            continue
                    
                    # Add content to the appropriate section
                    if current_section:
                        if sections[current_section]:
                            sections[current_section] += ' ' + line
                        else:
                            sections[current_section] = line
                
                # Build formatted text for plain text
                formatted_parts = []
                
                if sections['answer']:
                    formatted_parts.append(sections['answer'])
                
                if sections['source']:
                    sources = [s.strip() for s in sections['source'].split('\n') if s.strip()]
                    if sources:
                        formatted_parts.append(f"<br/><br/><b>Sources:</b><br/>• " + "<br/>• ".join(sources))
                
                if sections['tool_routing']:
                    formatted_parts.append(f"<br/><br/><b>Tool Selection & Query Routing:</b><br/>{sections['tool_routing']}")
                
                # If no structured sections found, treat as plain text
                if not any(sections.values()):
                    formatted_parts = [text]
                
                full_text = "".join(formatted_parts)
                full_text = re.sub(r'\s+', ' ', full_text)
                
                return full_text.strip()

        for i, message in enumerate(messages):
            role = message.get('role', '')
            content = message.get('content', '')
            
            # Convert markdown formatting to ReportLab-compatible format
            formatted_content = convert_markdown_to_reportlab(content)

            if role == 'user':
                story.append(Paragraph("Doctor Input:", section_header_style))
                story.append(Paragraph(formatted_content, content_style))
                story.append(Spacer(1, 8))
            elif role == 'ai':
                story.append(Paragraph("System Output:", section_header_style))
                story.append(Paragraph(formatted_content, content_style))
                story.append(Spacer(1, 8))
            else:
                # Generic role
                story.append(Paragraph(f"{role.title()}:", section_header_style))
                story.append(Paragraph(formatted_content, content_style))
                story.append(Spacer(1, 8))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_content = buffer.getvalue()

        # NEW: Upload to Azure if available
        azure_url = None
        if AZURE_AVAILABLE:
            try:
                # Prepare metadata
                metadata = {
                    'doctor_name': str(doctor_name),
                    'patient_name': str(patient_name) if patient_name else '',
                    'patient_id': str(patient_id) if patient_id else '',
                    'patient_problem': str(patient_problem) if patient_problem else '',
                    'pdf_type': 'research_chat',
                    'generated_at': datetime.now().isoformat(),
                    'message_count': str(len(messages)),
                    'has_json_data': str(bool(json_data))
                }

                # Upload to research container
                azure_url = upload_pdf_to_azure(pdf_content, filename, "research", metadata)

            except Exception as e:
                print(f"Azure upload failed: {e}")

        # Return PDF as response
        from flask import make_response
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'

        # Add Azure URL to response headers if available
        if azure_url:
            response.headers['X-Azure-URL'] = azure_url

        return response
        
    except ImportError:
        # Fallback if reportlab is not installed
        return jsonify({"error": "PDF generation not available. Please install reportlab."}), 500
    except Exception as e:
        print(f"Error generating chat PDF: {e}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500


@app.route("/generate_conversation_pdf", methods=["POST"])
def generate_conversation_pdf():
    """Generate PDF for conversation segments with voice diarization results"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.colors import black, green, blue
        import io
        import os
        
        data = request.json
        
        # Extract data
        doctor_name = data.get('doctorName', 'Doctor')
        patient_name = data.get('patientName', 'Patient')
        date_time = data.get('dateTime', '')
        segments = data.get('segments', [])
        full_transcript = data.get('fullTranscript', '')
        summary = data.get('summary', '')
        conclusion = data.get('conclusion', '')
        processing_info = data.get('processingInfo', {})
        is_duplicate = data.get('isDuplicate', False)
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch, leftMargin=inch, 
                               topMargin=inch, bottomMargin=inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=10,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            alignment=TA_CENTER,
            fontName='Helvetica',
            textColor=blue
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        segment_style = ParagraphStyle(
            'SegmentStyle',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            fontName='Helvetica'
        )
        
        doctor_segment_style = ParagraphStyle(
            'DoctorSegmentStyle',
            parent=segment_style,
            leftIndent=20,
            textColor=green
        )
        
        patient_segment_style = ParagraphStyle(
            'PatientSegmentStyle',
            parent=segment_style,
            leftIndent=20,
            textColor=blue
        )
        
        # Build PDF content
        story = []
        
        # Add logo and title
        logo_path = os.path.join(os.path.dirname(__file__), 'ClientLogo101.png')
        duplicate_text = " (Duplicate)" if is_duplicate else ""
        main_title = f"Doctor-Patient Conversation Analysis{duplicate_text}"
        
        if os.path.exists(logo_path):
            try:
                logo_img = Image(logo_path, width=1.5*inch, height=0.9*inch)
                title_paragraph = Paragraph(main_title, title_style)
                
                header_data = [[title_paragraph, logo_img]]
                header_table = Table(header_data, colWidths=[4.5*inch, 2*inch])
                header_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                story.append(header_table)
                story.append(Spacer(1, 15))
            except Exception as e:
                print(f"Warning: Could not add logo to PDF: {e}")
                story.append(Paragraph(main_title, title_style))
                story.append(Spacer(1, 15))
        else:
            story.append(Paragraph(main_title, title_style))
            story.append(Spacer(1, 15))
        
        # Add subtitle
        story.append(Paragraph("Voice Diarization & Transcription Results", subtitle_style))
        story.append(Spacer(1, 10))
        
        # Participant Information
        doctor_info = f"Doctor: {doctor_name}"
        patient_info = f"Patient: {patient_name}"
        
        story.append(Paragraph(doctor_info, styles['Normal']))
        story.append(Spacer(1, 5))
        story.append(Paragraph(patient_info, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Processing Information
        story.append(Paragraph("Processing Information", header_style))
        engine_info = processing_info.get('engine', 'OpenAI Voice Diarization + Whisper')
        total_segments = processing_info.get('totalSegments', len(segments))
        language_used = processing_info.get('language', 'English')
        was_translated = processing_info.get('translated', False)
        
        # Language display mapping
        language_display = {
            'en': 'English',
            'es': 'Spanish',
            'zh': 'Chinese',
            'yue': 'Cantonese',
            'tl': 'Tagalog',
            'hi': 'Hindi',
            'te': 'Telugu',
            'ta': 'Tamil',
            'gu': 'Gujarati',
            'pa': 'Punjabi'
        }
        language_name = language_display.get(language_used, language_used.title() if language_used else 'English')
        
        story.append(Paragraph("Voice Diarization: OpenAI-based Speaker Separation  Transcription Engine: Whisper", styles['Normal']))
        story.append(Paragraph(f"Language Majorly Spoken: {language_name}   Total Segments: {total_segments}", styles['Normal']))
        
        # Format date
        display_date = date_time if date_time else datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        story.append(Paragraph(f"Processing Date: {display_date}", styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Conversation Segments
        if segments and len(segments) > 0:
            story.append(Paragraph("Conversation Segments", header_style))
            
            for i, segment in enumerate(segments):
                role = segment.get('role', 'Unknown')
                text = segment.get('text', '')
                start_time = segment.get('start', '')
                end_time = segment.get('end', '')
                confidence = segment.get('confidence', 0)
                
                # Create segment header
                timing_info = f"[{start_time} - {end_time}]" if start_time and end_time else ""
                confidence_info = f"(Confidence: {int(confidence * 100)}%)" if confidence > 0 else ""
                header_text = f"{role} {timing_info} {confidence_info}"
                
                # Choose style based on role
                if role.lower() == 'doctor':
                    story.append(Paragraph(f"<b>{header_text}</b>", doctor_segment_style))
                    story.append(Paragraph(text, doctor_segment_style))
                else:  # Patient
                    story.append(Paragraph(f"<b>{header_text}</b>", patient_segment_style))
                    story.append(Paragraph(text, patient_segment_style))
                
                story.append(Spacer(1, 8))
            
            story.append(Spacer(1, 15))
        
        # Full Transcript
        story.append(Paragraph("Complete Transcript", header_style))
        transcript_text = full_transcript if full_transcript else "No transcript available"
        story.append(Paragraph(transcript_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Medical Summary
        if summary:
            story.append(Paragraph("Medical Summary", header_style))
            story.append(Paragraph(summary, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Conclusion & Recommendations
        if conclusion:
            story.append(Paragraph("Conclusion & Recommendations", header_style))
            story.append(Paragraph(conclusion, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        duplicate_suffix = "-DUPLICATE" if is_duplicate else ""
        filename = f"CONVERSATION-{doctor_name.upper().replace(' ', '')}{duplicate_suffix}-{timestamp}.pdf"
        
        # Upload to Azure if available
        azure_url = None
        if AZURE_AVAILABLE:
            try:
                metadata = {
                    'doctor_name': doctor_name,
                    'patient_name': patient_name,
                    'date_time': date_time,
                    'summary': summary if summary else '',
                    'conclusion': conclusion if conclusion else '',
                    'pdf_type': 'conversation_segments',
                    'is_duplicate': is_duplicate,
                    'total_segments': len(segments),
                    'generated_at': datetime.now().isoformat()
                }
                
                azure_url = upload_pdf_to_azure(pdf_content, filename, "conversation", metadata)
                
            except Exception as e:
                print(f"Azure upload failed: {e}")
        
        # Return PDF as response
        from flask import make_response
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        if azure_url:
            response.headers['X-Azure-URL'] = azure_url
        
        return response
        
    except ImportError:
        return jsonify({"error": "PDF generation not available. Please install reportlab."}), 500
    except Exception as e:
        print(f"Error generating conversation PDF: {e}")
        return jsonify({"error": f"Failed to generate conversation PDF: {str(e)}"}), 500


@app.route('/search_doctors', methods=['GET'])
def search_doctors():
    """
    Search for doctors by first_name and last_name from pces_users table.
    Returns matching doctors based on partial input.
    """
    try:
        query = request.args.get('q', '').strip().lower()
        if not query:
            return jsonify([])
        
        # Connect to database
        with _pg_conn() as conn:
            with conn.cursor() as cursor:
                # Search by first_name, last_name, or combined name
                search_query = """
                SELECT DISTINCT first_name, last_name 
                FROM pces_users 
                WHERE LOWER(first_name) LIKE %s 
                   OR LOWER(last_name) LIKE %s 
                   OR LOWER(CONCAT(first_name, ' ', last_name)) LIKE %s
                ORDER BY first_name, last_name
                LIMIT 10
                """
                
                search_pattern = f"%{query}%"
                cursor.execute(search_query, (search_pattern, search_pattern, search_pattern))
                results = cursor.fetchall()
                
                # Format results
                doctors = []
                for row in results:
                    if row[0] and row[1]:  # Ensure both names exist
                        full_name = f"{row[0]} {row[1]}"
                        doctors.append({
                            "first_name": row[0],
                            "last_name": row[1],
                            "full_name": full_name
                        })
                
                return jsonify(doctors)
                
    except Exception as e:
        print(f"Error searching doctors: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/search_patients', methods=['GET'])
def search_patients():
    """
    Search for patients by first_name and last_name from patient table.
    Returns matching patients based on partial input.
    """
    try:
        query = request.args.get('q', '').strip().lower()
        if not query:
            return jsonify([])
        
        # Connect to database
        with _pg_conn() as conn:
            with conn.cursor() as cursor:
                # Search by first_name, last_name, or combined name
                search_query = """
                SELECT DISTINCT patient_id, first_name, last_name 
                FROM patient 
                WHERE LOWER(first_name) LIKE %s 
                   OR LOWER(last_name) LIKE %s 
                   OR LOWER(CONCAT(first_name, ' ', last_name)) LIKE %s
                ORDER BY first_name, last_name
                LIMIT 10
                """
                
                search_pattern = f"%{query}%"
                cursor.execute(search_query, (search_pattern, search_pattern, search_pattern))
                results = cursor.fetchall()
                
                # Format results
                patients = []
                for row in results:
                    if row[1] and row[2]:  # Ensure both names exist
                        full_name = f"{row[1]} {row[2]}"
                        patients.append({
                            "patient_id": row[0],
                            "first_name": row[1],
                            "last_name": row[2],
                            "full_name": full_name
                        })
                
                return jsonify(patients)
                
    except Exception as e:
        print(f"Error searching patients: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe_doctor_patient_conversation", methods=["POST"])
def transcribe_doctor_patient_conversation():
    """Handle doctor-patient conversation recording with voice diarization and optional translation"""
    if not DIARIZATION_AVAILABLE:
        return jsonify({"error": "Voice diarization not available. Please install required dependencies."}), 500
    
    try:
        from langchain.schema import HumanMessage, SystemMessage
        
        audio_file = request.files['audio']
        doctor_name = request.form.get('doctor_name', 'Unknown Doctor')
        patient_name = request.form.get('patient_name', 'Unknown Patient')
        language = request.form.get('language', 'en')  # Get language parameter (default: English)
        
        print(f"Processing conversation for {doctor_name} and {patient_name} in language: {language}")
        
        # Save uploaded audio to temporary file (raw format from browser)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        # Convert audio to proper WAV format for processing
        try:
            from pydub import AudioSegment
            
            # Load audio file (supports WebM, MP3, WAV, etc.)
            print(f"Loading audio file: {temp_audio_path}")
            audio = AudioSegment.from_file(temp_audio_path)
            
            # Convert to mono 16kHz WAV (optimal for Whisper)
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Export as WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_file:
                audio.export(wav_file.name, format='wav')
                wav_audio_path = wav_file.name
            
            # Clean up original file
            os.unlink(temp_audio_path)
            temp_audio_path = wav_audio_path
            print(f"✅ Audio converted to WAV: {temp_audio_path}")
            
        except ImportError:
            # Fallback: Try soundfile for WAV files
            try:
                import soundfile as sf
                
                # Use soundfile directly (faster and no warnings)
                audio_data, sample_rate = sf.read(temp_audio_path)
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Resample if needed (using scipy for speed)
                if sample_rate != 16000:
                    from scipy import signal
                    num_samples = int(len(audio_data) * 16000 / sample_rate)
                    audio_data = signal.resample(audio_data, num_samples)
                    sample_rate = 16000
                
                # Create a new properly formatted WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as converted_audio:
                    sf.write(converted_audio.name, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                    converted_audio_path = converted_audio.name
                
                # Clean up original temp file
                os.unlink(temp_audio_path)
                temp_audio_path = converted_audio_path
                print(f"✅ Audio converted using soundfile: {temp_audio_path}")
                
            except Exception as e:
                print(f"⚠️ Audio conversion error: {e}")
                print("Warning: Using original audio format - may not work correctly")
        
        try:
            # Check if Spanish translation is needed
            if language == 'es':
                # Spanish mode: Directly use the working translate_audio logic
                print("🔄 Processing Spanish conversation - using translate_audio approach")
                
                # Create a mock request object to pass to translate_audio logic
                from werkzeug.datastructures import FileStorage
                from io import BytesIO
                
                # Read the audio file into memory
                with open(temp_audio_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Create mock files and form for internal processing
                mock_files = {'audio': FileStorage(stream=BytesIO(audio_bytes), filename='audio.wav')}
                mock_form = {'language': language}
                
                # Manually invoke translate_audio logic inline
                try:
                    # Transcribe with language hint
                    language_map = {
                        'es': 'spanish', 'zh': 'chinese', 'yue': 'chinese',
                        'tl': 'tagalog', 'hi': 'hindi', 'te': 'telugu',
                        'ta': 'tamil', 'gu': 'gujarati', 'pa': 'punjabi'
                    }
                    whisper_language = language_map.get(language, 'spanish')
                    
                    print(f"🎤 Step 1/3: Starting fast transcription in {whisper_language}...")
                    import time
                    start_time = time.time()
                    
                    # SPEED OPTIMIZATION: Use fp16, no_speech_threshold, and faster beam search
                    result = model.transcribe(
                        temp_audio_path, 
                        language=whisper_language,
                        fp16=device == "cuda",  # Use fp16 on GPU for 2x speed
                        beam_size=1,  # Faster beam search (greedy decoding)
                        best_of=1,  # Don't compare multiple samples
                        temperature=0.0,  # Deterministic output
                        condition_on_previous_text=False  # Faster processing
                    )
                    original_text = result['text']
                    transcription_time = time.time() - start_time
                    print(f"✅ Step 1/3 completed in {transcription_time:.2f}s. Text length: {len(original_text)} characters")
                    
                    print(f"🤖 Step 2/3: Starting AI segmentation and translation...")
                    segmentation_start = time.time()
                    
                    # Combined segmentation + translation (same as translate_audio)
                    combined_prompt = f"""Analyze this doctor-patient conversation in {whisper_language} and do TWO things:
1. Segment it by speaker (doctor vs patient) - EACH TURN/EXCHANGE should be a SEPARATE segment
2. Translate each segment to English

Transcript:
{original_text}

Provide your response in this exact JSON format:
{{
  "segments": [
    {{"speaker": "doctor", "text": "original text", "translated_text": "English translation"}},
    {{"speaker": "patient", "text": "original text", "translated_text": "English translation"}},
    {{"speaker": "doctor", "text": "original text", "translated_text": "English translation"}},
    {{"speaker": "patient", "text": "original text", "translated_text": "English translation"}}
  ]
}}

CRITICAL Rules:
- Split the conversation into INDIVIDUAL turns/exchanges - do NOT combine all doctor statements into one segment
- Each time the speaker changes, create a NEW segment
- Medical professionals use medical terminology, ask clinical questions, give advice
- Patients describe symptoms, ask questions about their health
- Provide accurate medical translations
- Be precise in segmentation - multiple back-and-forth exchanges should result in multiple segments"""
                    
                    messages = [
                        SystemMessage(content="You are an expert medical conversation analyzer and translator. Provide responses in valid JSON format only."),
                        HumanMessage(content=combined_prompt)
                    ]
                    
                    # SPEED OPTIMIZATION: Use faster LLM with optimized settings
                    from langchain_openai import ChatOpenAI
                    fast_llm = ChatOpenAI(
                        api_key=os.getenv("openai_api_key"),
                        base_url=os.getenv("base_url"),
                        model_name=os.getenv("llm_model_name"),
                        temperature=0.3,  # Faster inference with slight randomness
                        max_tokens=2000,  # Limit output length for speed
                        request_timeout=30
                    )
                    
                    response = fast_llm.invoke(messages)
                    result_content = response.content.strip()
                    
                    # Extract JSON from markdown if present
                    if "```json" in result_content:
                        result_content = result_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in result_content:
                        result_content = result_content.split("```")[1].split("```")[0].strip()
                    
                    segments_data = json.loads(result_content)
                    segments = segments_data.get('segments', [])
                    
                    # Ensure all segments have translated_text
                    for segment in segments:
                        if 'translated_text' not in segment:
                            segment['translated_text'] = segment.get('text', '')
                    
                    segmentation_time = time.time() - segmentation_start
                    print(f"✅ Step 2/3 completed in {segmentation_time:.2f}s. Processed {len(segments)} segments")
                    
                    print(f"📝 Step 3/3: Formatting conversation data...")
                    format_start = time.time()
                    
                    # Convert to conversation format
                    transcript = []
                    for i, segment in enumerate(segments):
                        role = segment.get('speaker', 'unknown').lower()
                        if role == 'doctor':
                            role = 'Doctor'
                        elif role == 'patient':
                            role = 'Patient'
                        else:
                            role = 'Unknown'
                        
                        transcript.append({
                            'role': role,
                            'text': segment.get('translated_text', segment.get('text', '')),
                            'start': f'{i * 10}s',
                            'end': f'{(i + 1) * 10}s',
                            'confidence': 0.95
                        })
                    
                    raw_transcript = '\n\n'.join([f"{seg.get('speaker', 'unknown').title()}: {seg.get('translated_text', seg.get('text', ''))}" for seg in segments])
                    
                    # Create original transcript in Spanish
                    original_transcript = '\n\n'.join([f"{seg.get('speaker', 'unknown').title()}: {seg.get('text', '')}" for seg in segments])
                    
                    format_time = time.time() - format_start
                    total_time = time.time() - start_time
                    print(f"✅ Step 3/3 completed in {format_time:.2f}s")
                    print(f"🎉 Total processing time: {total_time:.2f}s (Transcription: {transcription_time:.2f}s, Segmentation: {segmentation_time:.2f}s, Formatting: {format_time:.2f}s)")
                    
                    # Prepare conversation data
                    conversation_data = {
                        "doctor_name": doctor_name,
                        "patient_name": patient_name,
                        "session_date": datetime.now().isoformat(),
                        "duration": "Unknown",
                        "total_segments": len(transcript),
                        "doctor_segments": len([t for t in transcript if t.get('role', '').lower() == 'doctor']),
                        "patient_segments": len([t for t in transcript if t.get('role', '').lower() == 'patient']),
                        "speakers_detected": 2,
                        "transcript": transcript,
                        "raw_transcript": raw_transcript,
                        "original_transcript": original_transcript,
                        "role_mapping": {"Doctor": doctor_name, "Patient": patient_name},
                        "language": language,
                        "translated": True
                    }
                    
                except Exception as e:
                    print(f"Error in Spanish processing: {str(e)}")
                    traceback.print_exc()
                    return jsonify({"success": False, "error": f"Spanish translation failed: {str(e)}"}), 500
                
            else:
                # English mode: Use existing diarization processor
                print("🔄 Using OpenAI-only processing (bypassing pyannote)")
                diarization_processor = get_diarization_processor()
                
                # Run async function in thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    diarization_processor.process_doctor_patient_conversation_openai_only(temp_audio_path)
                )
                
                if result.get("error"):
                    return jsonify({"success": False, "error": result["error"]}), 500
                
                # Prepare conversation data
                conversation_data = {
                    "doctor_name": doctor_name,
                    "patient_name": patient_name,
                    "session_date": datetime.now().isoformat(),
                    "duration": result.get("total_duration", "Unknown"),
                    "total_segments": result.get("total_segments", 0),
                    "doctor_segments": result.get("doctor_segments", 0),
                    "patient_segments": result.get("patient_segments", 0),
                    "speakers_detected": result.get("speakers_detected", 0),
                    "transcript": result.get("transcript", []),
                    "raw_transcript": result.get("raw_transcript", ""),
                    "original_transcript": result.get("raw_transcript", ""),
                    "role_mapping": result.get("role_mapping", {}),
                    "language": language,
                    "translated": False
                }
            
            # Generate medical summary and conclusion from the conversation transcript
            full_transcript = conversation_data.get("raw_transcript", "")
            if full_transcript:
                try:
                    print("Generating medical summary and conclusion for conversation...")
                    
                    summary_prompt = f"""
                    As a medical AI assistant, please analyze the following doctor-patient conversation transcript and provide a professional medical summary.
                    
                    Doctor: {doctor_name}
                    Patient: {patient_name}
                    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    Conversation Transcript:
                    {full_transcript}
                    
                    Please provide:
                    1. A concise clinical summary highlighting key medical information, symptoms, findings, and discussions from the conversation
                    2. Professional conclusions with recommendations, follow-up actions, or treatment plans mentioned during the conversation
                    
                    Format your response exactly as:
                    SUMMARY:
                    [Provide a clear, professional summary of the medical conversation]
                    
                    CONCLUSION:
                    [Provide conclusions, recommendations, and any follow-up actions mentioned]
                    """
                    
                    # Get AI response
                    ai_response = llm.invoke(summary_prompt)
                    if hasattr(ai_response, 'content'):
                        ai_content = ai_response.content.strip()
                    else:
                        ai_content = str(ai_response).strip()
                    
                    print(f"AI summary generated for conversation. Length: {len(ai_content)} characters")
                    
                    # Parse summary and conclusion from AI response
                    summary_parts = ai_content.split("CONCLUSION:")
                    if len(summary_parts) == 2:
                        summary = summary_parts[0].replace("SUMMARY:", "").strip()
                        conclusion = summary_parts[1].strip()
                    else:
                        # Fallback parsing method
                        lines = ai_content.split('\n')
                        summary_lines = []
                        conclusion_lines = []
                        in_conclusion = False
                        
                        for line in lines:
                            if 'CONCLUSION' in line.upper():
                                in_conclusion = True
                                continue
                            elif 'SUMMARY' in line.upper():
                                in_conclusion = False
                                continue
                            
                            if in_conclusion:
                                conclusion_lines.append(line)
                            else:
                                summary_lines.append(line)
                        
                        summary = '\n'.join(summary_lines).strip()
                        conclusion = '\n'.join(conclusion_lines).strip()
                        
                        # Final fallback
                        if not summary and not conclusion:
                            summary = "Medical conversation analysis completed. Please refer to the full transcript for detailed information."
                            conclusion = "Further medical review and assessment may be required based on the conversation content."
                    
                    # Add summary and conclusion to conversation data
                    conversation_data["summary"] = summary
                    conversation_data["conclusion"] = conclusion
                    
                except Exception as e:
                    print(f"Error generating summary for conversation: {str(e)}")
                    # Add default summary and conclusion if generation fails
                    conversation_data["summary"] = "Summary generation was not available for this conversation."
                    conversation_data["conclusion"] = "Please review the conversation transcript for medical conclusions."
            else:
                # Add default summary and conclusion if no transcript
                conversation_data["summary"] = "No transcript available for summary generation."
                conversation_data["conclusion"] = "Please ensure audio quality is sufficient for transcription."
            
            return jsonify({
                "success": True,
                "conversation_data": conversation_data,
                "message": "Doctor-patient conversation processed successfully"
            })
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
                
    except Exception as e:
        print(f"Error processing doctor-patient conversation: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Update existing PDF generation functions to include Azure uploads
def upload_pdf_to_azure(pdf_content, filename, pdf_type, metadata=None):
    """Helper function to upload PDFs to Azure"""
    if not AZURE_AVAILABLE:
        print("Azure storage not available")
        return None
    
    try:
        storage_manager = get_storage_manager()
        
        if pdf_type == "research":
            return storage_manager.upload_research_pdf(pdf_content, filename, metadata.get('patient_problem'), metadata)
        elif pdf_type == "patient_summary":
            # Extract patient data from metadata for the patient_data parameter
            patient_data = {
                'patient_name': metadata.get('patient_name'),
                'patient_id': metadata.get('patient_id'),
                'doctor_name': metadata.get('doctor_name'),
                'session_date': metadata.get('date_time')
            }
            return storage_manager.upload_patient_summary_pdf(pdf_content, filename, patient_data, metadata)
        elif pdf_type == "conversation":
            # Extract conversation data from metadata for the conversation_data parameter
            conversation_data = {
                'doctor_name': metadata.get('doctor_name'),
                'patient_name': metadata.get('patient_name'),
                'duration': metadata.get('duration', 'Unknown'),
                'session_date': metadata.get('date_time')
            }
            return storage_manager.upload_conversation_pdf(pdf_content, filename, conversation_data, metadata)
        
    except Exception as e:
        print(f"Azure upload failed: {e}")
        return None


@app.route('/check_azure_files', methods=['GET'])
def check_azure_files():
    """Check what files have been uploaded to Azure"""
    if not AZURE_AVAILABLE:
        return jsonify({"error": "Azure storage not available"}), 500
    
    try:
        storage_manager = get_storage_manager()
        
        # Get file type filter from query params
        file_type = request.args.get('type', 'all')  # 'research', 'patient_summary', 'conversation', or 'all'
        
        files_info = {}
        
        if file_type in ['research', 'all']:
            research_files = storage_manager.list_files_in_container("contoso", "pces/documents/research/")
            files_info['research_files'] = research_files
        
        if file_type in ['patient_summary', 'all']:
            patient_files = storage_manager.list_files_in_container("contoso", "pces/documents/doc-patient-summary/")
            files_info['patient_summary_files'] = patient_files
        
        if file_type in ['conversation', 'all']:
            conversation_files = storage_manager.list_files_in_container("contoso", "pces/documents/conversation/")
            files_info['conversation_files'] = conversation_files
        
        # Count totals
        total_files = sum(len(files) for files in files_info.values())
        
        return jsonify({
            "success": True,
            "total_files": total_files,
            "files": files_info,
            "message": f"Found {total_files} files in Azure storage"
        })
        
    except Exception as e:
        print(f"Error checking Azure files: {e}")
        return jsonify({"error": f"Failed to check Azure files: {str(e)}"}), 500


@app.route('/check_azure_file/<path:filename>', methods=['GET'])
def check_azure_file(filename):
    """Check if a specific file exists in Azure"""
    if not AZURE_AVAILABLE:
        return jsonify({"error": "Azure storage not available"}), 500
    
    try:
        storage_manager = get_storage_manager()
        container_name = request.args.get('container', 'contoso')
        file_path = request.args.get('path', f'pces/documents/research/{filename}')
        
        # Check if file exists
        exists = storage_manager.check_file_exists(container_name, file_path)
        
        if exists:
            # Get file metadata
            file_info = storage_manager.get_file_metadata(container_name, file_path)
            return jsonify({
                "success": True,
                "exists": True,
                "file_info": file_info
            })
        else:
            return jsonify({
                "success": True,
                "exists": False,
                "message": f"File {filename} not found in Azure"
            })
        
    except Exception as e:
        print(f"Error checking Azure file: {e}")
        return jsonify({"error": f"Failed to check Azure file: {str(e)}"}), 500


@app.route('/azure_storage_info', methods=['GET'])
def azure_storage_info():
    """Get Azure storage configuration and status"""
    try:
        info = {
            "azure_available": AZURE_AVAILABLE,
            "connection_configured": bool(os.getenv('AZURE_STORAGE_CONNECTION_STRING')),
            "containers": {
                "contoso": {
                    "research_path": "pces/documents/research/",
                    "patient_summary_path": "pces/documents/doc-patient-summary/",
                    "conversations_path": "pces/documents/conversation/"
                }
            }
        }
        
        if AZURE_AVAILABLE and info["connection_configured"]:
            try:
                storage_manager = get_storage_manager()
                # Test connection by listing containers
                containers = []
                for container in storage_manager.blob_service_client.list_containers():
                    containers.append(container.name)
                info["available_containers"] = containers
                info["connection_status"] = "Connected"
            except Exception as e:
                info["connection_status"] = f"Connection failed: {str(e)}"
        else:
            info["connection_status"] = "Not configured"
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": f"Failed to get Azure info: {str(e)}"}), 500


# ============================================
# RLHF (Reinforcement Learning from Human Feedback) Admin Routes
# ============================================

@app.route('/admin/rlhf')
def admin_rlhf():
    """Render the RLHF admin page"""
    return render_template('admin_rlhf.html')


@app.route('/api/rlhf/stats', methods=['GET'])
def get_rlhf_stats():
    """Get statistics about RLHF training data"""
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                # Get total interactions
                cur.execute("SELECT COUNT(*) FROM rlhf_interactions")
                total_interactions = cur.fetchone()[0]
                
                # Get average rating
                cur.execute("SELECT AVG(rating) FROM rlhf_interactions WHERE rating IS NOT NULL")
                avg_rating_result = cur.fetchone()[0]
                avg_rating = float(avg_rating_result) if avg_rating_result else 0.0
                
                # Get total sessions
                cur.execute("SELECT COUNT(*) FROM rlhf_sessions")
                total_sessions = cur.fetchone()[0]
                
                # Get bias count
                cur.execute("SELECT COUNT(*) FROM rlhf_interactions WHERE bias_flag = TRUE")
                bias_count = cur.fetchone()[0]
                
                return jsonify({
                    "total_interactions": total_interactions,
                    "avg_rating": avg_rating,
                    "total_sessions": total_sessions,
                    "bias_count": bias_count
                })
    except Exception as e:
        print(f"Error getting RLHF stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rlhf/interactions', methods=['GET'])
def get_rlhf_interactions():
    """Get RLHF interactions with optional filters"""
    try:
        session_id = request.args.get('session_id', type=int)
        min_rating = request.args.get('min_rating', type=int)
        bias_only = request.args.get('bias_only', 'false').lower() == 'true'
        limit = request.args.get('limit', type=int, default=50)
        
        query = """
            SELECT interaction_id, session_id, user_prompt, ai_response, 
                   rating, feedback_comment, bias_flag, created_dt, updated_dt
            FROM rlhf_interactions
            WHERE 1=1
        """
        params = []
        
        if session_id:
            query += " AND session_id = %s"
            params.append(session_id)
        
        if min_rating:
            query += " AND rating >= %s"
            params.append(min_rating)
        
        if bias_only:
            query += " AND bias_flag = TRUE"
        
        query += " ORDER BY created_dt DESC LIMIT %s"
        params.append(limit)
        
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                results = cur.fetchall()
                
                interactions = []
                for row in results:
                    interaction = dict(zip(columns, row))
                    # Convert datetime to string for JSON serialization
                    if interaction.get('created_dt'):
                        interaction['created_dt'] = interaction['created_dt'].isoformat()
                    if interaction.get('updated_dt'):
                        interaction['updated_dt'] = interaction['updated_dt'].isoformat()
                    interactions.append(interaction)
                
                return jsonify(interactions)
    except Exception as e:
        print(f"Error getting RLHF interactions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rlhf/sessions', methods=['GET'])
def get_rlhf_sessions():
    """Get all RLHF training sessions"""
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, user_id, model_version, session_start, 
                           session_end, status, notes, created_dt
                    FROM rlhf_sessions
                    ORDER BY session_start DESC
                """)
                columns = [desc[0] for desc in cur.description]
                results = cur.fetchall()
                
                sessions = []
                for row in results:
                    session = dict(zip(columns, row))
                    # Convert datetime to string for JSON serialization
                    if session.get('session_start'):
                        session['session_start'] = session['session_start'].isoformat()
                    if session.get('session_end'):
                        session['session_end'] = session['session_end'].isoformat()
                    if session.get('created_dt'):
                        session['created_dt'] = session['created_dt'].isoformat()
                    sessions.append(session)
                
                return jsonify(sessions)
    except Exception as e:
        print(f"Error getting RLHF sessions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rlhf/add_sample', methods=['POST'])
def add_rlhf_sample():
    """Add a new RLHF training sample"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['session_id', 'user_prompt', 'ai_response']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rlhf_interactions 
                    (session_id, user_prompt, ai_response, rating, feedback_comment, 
                     bias_flag, created_by, updated_by, created_dt, updated_dt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING interaction_id
                """, (
                    data['session_id'],
                    data['user_prompt'],
                    data['ai_response'],
                    data.get('rating', 3),
                    data.get('feedback_comment', ''),
                    data.get('bias_flag', False),
                    1001,  # Default admin user
                    1001
                ))
                interaction_id = cur.fetchone()[0]
                conn.commit()
                
                return jsonify({
                    "success": True,
                    "interaction_id": interaction_id,
                    "message": "Training sample added successfully"
                })
    except Exception as e:
        print(f"Error adding RLHF sample: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rlhf/generate_samples', methods=['POST'])
def generate_rlhf_samples():
    """Generate multiple sample RLHF training interactions"""
    try:
        # Sample medical training data
        samples = [
            {
                "session_id": 1,
                "user_prompt": "What are the current guidelines for managing Type 2 Diabetes?",
                "ai_response": "According to ADA 2024 guidelines, Type 2 Diabetes management includes: 1) Metformin as first-line therapy, 2) HbA1c target <7% for most adults, 3) SGLT2 inhibitors or GLP-1 agonists for cardiovascular benefits, 4) Lifestyle modifications with diet and exercise.",
                "rating": 5,
                "feedback_comment": "Accurate and comprehensive response with latest guidelines",
                "bias_flag": False
            },
            {
                "session_id": 1,
                "user_prompt": "What is the recommended treatment for community-acquired pneumonia?",
                "ai_response": "For outpatient CAP treatment: 1) Amoxicillin 1g TID or Doxycycline 100mg BID for 5-7 days, 2) Add macrolide if atypical pathogens suspected, 3) Consider comorbidities and local resistance patterns.",
                "rating": 4,
                "feedback_comment": "Good response but could mention severity assessment criteria",
                "bias_flag": False
            },
            {
                "session_id": 2,
                "user_prompt": "How do I interpret elevated troponin levels?",
                "ai_response": "Elevated troponin indicates myocardial injury. Consider: 1) Acute MI (most common), 2) Myocarditis, 3) PE, 4) Renal failure, 5) Sepsis. Must correlate with clinical presentation, ECG changes, and cardiac imaging.",
                "rating": 5,
                "feedback_comment": "Excellent differential diagnosis approach",
                "bias_flag": False
            },
            {
                "session_id": 2,
                "user_prompt": "What are the indications for starting statin therapy?",
                "ai_response": "ACC/AHA 2023 guidelines recommend statins for: 1) LDL-C ≥190 mg/dL, 2) Diabetes (age 40-75) with LDL 70-189 mg/dL, 3) 10-year ASCVD risk ≥7.5%, 4) Clinical ASCVD.",
                "rating": 5,
                "feedback_comment": "Up-to-date guidelines accurately cited",
                "bias_flag": False
            },
            {
                "session_id": 3,
                "user_prompt": "What is the first-line treatment for COPD exacerbation?",
                "ai_response": "GOLD 2024 guidelines: 1) Short-acting bronchodilators (SABA/SAMA), 2) Systemic corticosteroids (Prednisone 40mg x 5 days), 3) Antibiotics if bacterial infection suspected (increased sputum purulence).",
                "rating": 4,
                "feedback_comment": "Correct but could elaborate on antibiotic selection",
                "bias_flag": False
            },
            {
                "session_id": 3,
                "user_prompt": "When should I order a CT scan for suspected stroke?",
                "ai_response": "Non-contrast CT head should be performed immediately for any suspected stroke to: 1) Rule out hemorrhage, 2) Exclude stroke mimics, 3) Determine eligibility for thrombolysis. CT angiography if large vessel occlusion suspected.",
                "rating": 5,
                "feedback_comment": "Critical time-sensitive information accurately provided",
                "bias_flag": False
            },
            {
                "session_id": 4,
                "user_prompt": "What are the criteria for diagnosing rheumatoid arthritis?",
                "ai_response": "ACR/EULAR 2010 criteria require ≥6 points from: 1) Joint involvement (0-5 points), 2) Serology (RF/anti-CCP, 0-3 points), 3) Acute phase reactants (ESR/CRP, 0-1 point), 4) Duration ≥6 weeks (1 point).",
                "rating": 5,
                "feedback_comment": "Comprehensive diagnostic criteria well explained",
                "bias_flag": False
            },
            {
                "session_id": 4,
                "user_prompt": "How do you manage acute asthma exacerbation?",
                "ai_response": "GINA 2024 approach: 1) High-dose SABA via spacer/nebulizer, 2) Oxygen to maintain SpO2 93-95%, 3) Systemic corticosteroids early, 4) Consider ipratropium bromide, 5) Assess severity and response continuously.",
                "rating": 4,
                "feedback_comment": "Good management outline, could add severity classification",
                "bias_flag": False
            },
            {
                "session_id": 5,
                "user_prompt": "What are the red flags for low back pain?",
                "ai_response": "Red flags requiring urgent evaluation: 1) Age <20 or >55, 2) History of cancer, 3) Constitutional symptoms, 4) Trauma, 5) Neurological deficits, 6) Saddle anesthesia, 7) Bowel/bladder dysfunction.",
                "rating": 5,
                "feedback_comment": "Essential red flags comprehensively listed",
                "bias_flag": False
            },
            {
                "session_id": 5,
                "user_prompt": "What immunizations does an adult with diabetes need?",
                "ai_response": "CDC recommendations for adults with diabetes: 1) Annual influenza vaccine, 2) Pneumococcal vaccines (PCV15/PCV20 or PPSV23), 3) Hepatitis B series, 4) Tdap/Td boosters, 5) COVID-19 vaccines, 6) Shingles vaccine (age ≥50).",
                "rating": 5,
                "feedback_comment": "Complete immunization schedule for diabetic patients",
                "bias_flag": False
            }
        ]
        
        inserted_count = 0
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                for sample in samples:
                    cur.execute("""
                        INSERT INTO rlhf_interactions 
                        (session_id, user_prompt, ai_response, rating, feedback_comment, 
                         bias_flag, created_by, updated_by, created_dt, updated_dt)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """, (
                        sample['session_id'],
                        sample['user_prompt'],
                        sample['ai_response'],
                        sample['rating'],
                        sample['feedback_comment'],
                        sample['bias_flag'],
                        1001,
                        1001
                    ))
                    inserted_count += 1
                conn.commit()
        
        return jsonify({
            "success": True,
            "count": inserted_count,
            "message": f"Successfully generated {inserted_count} training samples"
        })
    except Exception as e:
        print(f"Error generating RLHF samples: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rlhf/update_rating', methods=['POST'])
def update_rlhf_rating():
    """Update rating for an existing interaction"""
    try:
        data = request.json
        
        if 'interaction_id' not in data or 'rating' not in data:
            return jsonify({"error": "Missing interaction_id or rating"}), 400
        
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE rlhf_interactions 
                    SET rating = %s, 
                        feedback_comment = %s,
                        bias_flag = %s,
                        updated_by = %s,
                        updated_dt = CURRENT_TIMESTAMP
                    WHERE interaction_id = %s
                """, (
                    data['rating'],
                    data.get('feedback_comment', ''),
                    data.get('bias_flag', False),
                    1001,
                    data['interaction_id']
                ))
                conn.commit()
                
                return jsonify({
                    "success": True,
                    "message": "Rating updated successfully"
                })
    except Exception as e:
        print(f"Error updating RLHF rating: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rlhf/update_interaction', methods=['POST'])
def update_rlhf_interaction():
    """Update an entire RLHF interaction (prompt, response, rating, feedback, bias)"""
    try:
        data = request.json
        
        if 'interaction_id' not in data:
            return jsonify({"error": "Missing interaction_id"}), 400
        
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE rlhf_interactions 
                    SET user_prompt = %s,
                        ai_response = %s,
                        rating = %s, 
                        feedback_comment = %s,
                        bias_flag = %s,
                        updated_by = %s,
                        updated_dt = CURRENT_TIMESTAMP
                    WHERE interaction_id = %s
                """, (
                    data.get('user_prompt'),
                    data.get('ai_response'),
                    data.get('rating', 3),
                    data.get('feedback_comment', ''),
                    data.get('bias_flag', False),
                    1001,
                    data['interaction_id']
                ))
                conn.commit()
                
                return jsonify({
                    "success": True,
                    "message": "Interaction updated successfully"
                })
    except Exception as e:
        print(f"Error updating RLHF interaction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rlhf/delete_interaction', methods=['POST'])
def delete_rlhf_interaction():
    """Delete an RLHF interaction"""
    try:
        data = request.json
        
        if 'interaction_id' not in data:
            return jsonify({"error": "Missing interaction_id"}), 400
        
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM rlhf_interactions 
                    WHERE interaction_id = %s
                """, (data['interaction_id'],))
                conn.commit()
                
                return jsonify({
                    "success": True,
                    "message": "Interaction deleted successfully"
                })
    except Exception as e:
        print(f"Error deleting RLHF interaction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rlhf/update_session', methods=['POST'])
def update_rlhf_session():
    """Update an RLHF session"""
    try:
        data = request.json
        
        if 'session_id' not in data:
            return jsonify({"error": "Missing session_id"}), 400
        
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE rlhf_sessions 
                    SET user_id = %s,
                        model_version = %s,
                        status = %s,
                        notes = %s,
                        updated_by = %s,
                        updated_dt = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                """, (
                    data.get('user_id'),
                    data.get('model_version'),
                    data.get('status'),
                    data.get('notes', ''),
                    1001,
                    data['session_id']
                ))
                conn.commit()
                
                return jsonify({
                    "success": True,
                    "message": "Session updated successfully"
                })
    except Exception as e:
        print(f"Error updating RLHF session: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rlhf/train_model', methods=['POST'])
def train_rlhf_model():
    """
    Train the RLHF reward model using current feedback data.
    This endpoint runs the training pipeline and returns results.
    """
    try:
        import subprocess
        import threading
        import time
        from datetime import datetime
        
        # Check if there's enough training data
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM rlhf_interactions WHERE rating IS NOT NULL")
                rated_count = cur.fetchone()[0]
        
        min_samples = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "20"))
        
        if rated_count < min_samples:
            return jsonify({
                "success": False,
                "error": f"Insufficient training data. Found {rated_count} rated samples, need at least {min_samples}.",
                "rated_count": rated_count,
                "min_required": min_samples
            }), 400
        
        # Run training in a separate thread to avoid blocking
        training_output = {"status": "running", "output": "", "error": None}
        
        def run_training():
            try:
                # Run the training script
                result = subprocess.run(
                    ["python", "train_reward_sbert.py"],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                training_output["status"] = "completed" if result.returncode == 0 else "failed"
                training_output["output"] = result.stdout
                training_output["error"] = result.stderr if result.returncode != 0 else None
                training_output["return_code"] = result.returncode
                
            except subprocess.TimeoutExpired:
                training_output["status"] = "timeout"
                training_output["error"] = "Training exceeded 5 minute timeout"
            except Exception as e:
                training_output["status"] = "error"
                training_output["error"] = str(e)
        
        # Start training in background thread
        thread = threading.Thread(target=run_training)
        thread.start()
        
        # Wait up to 120 seconds for training to complete
        thread.join(timeout=120)
        
        if thread.is_alive():
            # Training still running after 2 minutes
            return jsonify({
                "success": False,
                "error": "Training is taking longer than expected. Please check server logs.",
                "status": "timeout"
            }), 408
        
        # Training completed
        if training_output["status"] == "completed":
            # Parse training output for metrics
            output_lines = training_output["output"]
            
            # Extract key metrics from output
            accuracy = None
            auc = None
            total_samples = rated_count
            
            for line in output_lines.split('\n'):
                if 'Accuracy:' in line:
                    try:
                        accuracy = float(line.split('Accuracy:')[1].strip())
                    except:
                        pass
                if 'AUC-ROC:' in line or 'AUC:' in line:
                    try:
                        auc = float(line.split(':')[1].strip())
                    except:
                        pass
            
            return jsonify({
                "success": True,
                "message": "Model trained successfully!",
                "metrics": {
                    "total_samples": total_samples,
                    "accuracy": accuracy,
                    "auc": auc,
                    "trained_at": datetime.now().isoformat()
                },
                "output": output_lines[-500:] if len(output_lines) > 500 else output_lines  # Last 500 chars
            })
        else:
            return jsonify({
                "success": False,
                "error": training_output.get("error", "Training failed"),
                "status": training_output["status"],
                "output": training_output.get("output", "")
            }), 500
            
    except Exception as e:
        print(f"Error training RLHF model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/rlhf/training_status', methods=['GET'])
def get_training_status():
    """
    Get current training model status and statistics.
    """
    try:
        import os
        from datetime import datetime
        
        # Check if model file exists
        model_path = os.getenv("REWARD_MODEL_PATH", "reward_model.joblib")
        model_exists = os.path.exists(model_path)
        
        # Get model file timestamp if it exists
        model_info = None
        if model_exists:
            stat = os.stat(model_path)
            model_info = {
                "exists": True,
                "path": model_path,
                "size_kb": round(stat.st_size / 1024, 2),
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        
        # Get training data count
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM rlhf_interactions WHERE rating IS NOT NULL")
                rated_count = cur.fetchone()[0]
                
                # Get latest training run from database if table exists
                training_history = None
                try:
                    cur.execute("""
                        SELECT model_version, total_interactions, accuracy, avg_reward, created_dt 
                        FROM rlhf_reward_model_training 
                        ORDER BY created_dt DESC 
                        LIMIT 1
                    """)
                    latest_training = cur.fetchone()
                    
                    if latest_training:
                        training_history = {
                            "model_version": latest_training[0],
                            "total_interactions": latest_training[1],
                            "accuracy": latest_training[2],
                            "avg_reward": latest_training[3],
                            "trained_at": latest_training[4].isoformat() if latest_training[4] else None
                        }
                        print(f"✅ Found training history: {training_history}")
                    else:
                        print("⚠️ No training history found in database")
                except Exception as e:
                    print(f"⚠️ Error fetching training history: {e}")
                    training_history = None
        
        min_samples = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "20"))
        
        return jsonify({
            "success": True,
            "model": model_info,
            "training_data": {
                "rated_count": rated_count,
                "min_required": min_samples,
                "ready_to_train": rated_count >= min_samples
            },
            "latest_training": training_history
        })
        
    except Exception as e:
        print(f"Error getting training status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/rlhf/model_info', methods=['GET'])
def get_model_info():
    """
    Get information about the RLHF reward model.
    """
    try:
        from rlhf_reranker import get_model_info, is_model_ready
        
        model_ready = is_model_ready()
        info = get_model_info()
        
        return jsonify({
            "success": True,
            "model_ready": model_ready,
            **info
        })
    except Exception as e:
        print(f"Error getting model info: {e}")
        return jsonify({
            "success": False,
            "model_ready": False,
            "error": str(e),
            "message": "Model not available. Please train the model first."
        })


@app.route('/api/rlhf/score', methods=['POST'])
def score_answer():
    """
    Score a single prompt-answer pair using the RLHF reward model.
    
    Request body:
        {
            "prompt": "What are the symptoms of pneumonia?",
            "answer": "Pneumonia symptoms include..."
        }
    
    Returns:
        {
            "success": true,
            "score": 0.8734,
            "prompt": "...",
            "answer": "..."
        }
    """
    try:
        data = request.json
        prompt = data.get('prompt', '')
        answer = data.get('answer', '')
        
        if not prompt or not answer:
            return jsonify({
                "success": False,
                "error": "Both 'prompt' and 'answer' are required"
            }), 400
        
        from rlhf_reranker import score_text_pair, is_model_ready
        
        if not is_model_ready():
            return jsonify({
                "success": False,
                "error": "Model not available. Please train the model first."
            }), 503
        
        score = score_text_pair(prompt, answer)
        
        return jsonify({
            "success": True,
            "score": float(score),
            "prompt": prompt,
            "answer": answer
        })
        
    except Exception as e:
        print(f"Error scoring answer: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/rlhf/generate_candidates', methods=['POST'])
def generate_candidates():
    """
    Generate multiple candidate answers for a given question using LLM.
    Creates high-quality, medium-quality, and low-quality responses for testing.
    
    Request body:
        {
            "prompt": "What medications are used to treat asthma?"
        }
    
    Returns:
        {
            "success": true,
            "candidates": [
                "Comprehensive answer...",
                "Adequate answer...",
                "Brief answer..."
            ]
        }
    """
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": "'prompt' is required"
            }), 400
        
        from langchain.schema import HumanMessage, SystemMessage
        
        # Generate three different quality levels of answers
        generation_prompt = f"""Given this medical question: "{prompt}"

Generate THREE different quality answer variations for testing an AI quality assessment model:

1. HIGH-QUALITY ANSWER (5/5 stars): 
   - Comprehensive and detailed
   - Evidence-based with specific medication names, mechanisms, dosages
   - Includes treatment guidelines and clinical recommendations
   - Well-structured and thorough
   
2. MEDIUM-QUALITY ANSWER (3/5 stars):
   - Covers main points adequately
   - Mentions key medications but with less detail
   - Some specifics but not comprehensive
   - Acceptable but not exceptional
   
3. LOW-QUALITY ANSWER (1-2/5 stars):
   - Very brief and incomplete
   - Vague or overly general
   - Missing important details
   - May contain minimal useful information

Provide ONLY the three answers without any labels, numbering, or extra commentary. Separate each answer with "|||" delimiter.

Format: [Answer 1]|||[Answer 2]|||[Answer 3]"""
        
        # Use the LLM to generate candidates with a simpler, more direct prompt
        simple_prompt = f"""Question: {prompt}

Generate exactly 3 answers of varying quality:

Answer 1:
[Write a detailed, comprehensive answer with specific medications, dosages, and treatment guidelines]

Answer 2:
[Write a good but less detailed answer covering main points]

Answer 3:
[Write a very brief, vague answer]"""
        
        response = llm.invoke([
            SystemMessage(content=(
                "You are a medical expert writing clinical answer variations for evaluation. "
                "IMPORTANT: Write ONLY in plain prose. Do NOT include Python code, code blocks, "
                "programming exercises, markdown headers, or tutorial-style content. "
                "Every answer must be a natural clinical response a doctor would give to a patient."
            )),
            HumanMessage(content=simple_prompt)
        ])
        
        # Parse the response
        response_text = response.content.strip()
        print(f"DEBUG - Raw LLM Response:\n{response_text}\n")
        
        candidates = []
        
        # Try multiple parsing strategies
        # Strategy 1: Split by "Answer X" markers
        import re
        answer_pattern = r'Answer\s+(\d+)[:\s\(]'
        matches = list(re.finditer(answer_pattern, response_text, re.IGNORECASE))
        
        if len(matches) >= 2:
            for i in range(len(matches)):
                start_pos = matches[i].end()
                end_pos = matches[i+1].start() if i+1 < len(matches) else len(response_text)
                answer_text = response_text[start_pos:end_pos].strip()
                # Clean up
                answer_text = re.sub(r'^\[.*?\]\s*', '', answer_text)  # Remove [description]
                answer_text = answer_text.split('\n\n')[0].strip()  # Take first paragraph
                if answer_text and len(answer_text) > 10:
                    candidates.append(answer_text)
        
        # Strategy 2: If not enough candidates, try ||| delimiter
        if len(candidates) < 3 and "|||" in response_text:
            candidates = [ans.strip() for ans in response_text.split("|||") if ans.strip()]
        
        # Strategy 3: Split by double newlines and filter
        if len(candidates) < 3:
            paragraphs = [p.strip() for p in response_text.split("\n\n") if p.strip() and len(p.strip()) > 20]
            # Remove lines that are just headers/labels
            filtered = []
            for p in paragraphs:
                if not re.match(r'^(Answer\s+\d+|HIGH|MEDIUM|LOW|Question:)', p, re.IGNORECASE):
                    # Clean up any inline labels
                    p = re.sub(r'^(Answer\s+\d+[:\s\(].*?[\):]?\s*)', '', p, flags=re.IGNORECASE)
                    if p.strip() and len(p.strip()) > 10:
                        filtered.append(p.strip())
            candidates = filtered[:3]
        
        print(f"DEBUG - Parsed {len(candidates)} candidates")
        for i, c in enumerate(candidates):
            print(f"Candidate {i+1}: {c[:100]}...")
        
        # Ensure we have exactly 3 candidates with quality-appropriate fallbacks
        if len(candidates) < 1:
            candidates.append(f"Comprehensive treatment for {prompt.lower().replace('what', '').replace('?', '').strip()} includes multiple evidence-based approaches with specific medications and clinical guidelines.")
        if len(candidates) < 2:
            candidates.append(f"Treatment typically involves standard medications and therapies.")
        if len(candidates) < 3:
            candidates.append(f"Use medications.")
        
        # Clean up any remaining formatting artifacts
        cleaned_candidates = []
        for candidate in candidates[:3]:
            # Remove code blocks (``` ... ```) — never appropriate in clinical answers
            candidate = re.sub(r'```[\s\S]*?```', '', candidate)
            # Remove inline code
            candidate = re.sub(r'`[^`]+`', '', candidate)
            # Remove Exercise / Ideas / Solution sections that tutorials include
            candidate = re.sub(r'\n*(#+\s*)?(Exercise|Ideas|Solution|Example|import|def |print\()[^\n]*(\n[^\n]+)*', '', candidate, flags=re.IGNORECASE)
            # Remove markdown bold/italic/headers
            candidate = re.sub(r'^\*\*.*?\*\*\s*', '', candidate)
            candidate = re.sub(r'^#+\s+', '', candidate)
            # Collapse excessive whitespace
            candidate = re.sub(r'\n{3,}', '\n\n', candidate)
            candidate = candidate.strip()
            if candidate:
                cleaned_candidates.append(candidate)
        
        candidates = cleaned_candidates
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "candidates": candidates[:3]  # Return exactly 3
        })
        
    except Exception as e:
        print(f"Error generating candidates: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/rlhf/rerank', methods=['POST'])
def rerank_answers():
    """
    Re-rank multiple candidate answers using the RLHF reward model.
    
    Request body:
        {
            "prompt": "What medications treat asthma?",
            "candidates": [
                "Inhalers like albuterol...",
                "Long-term control medications...",
                "Try breathing exercises..."
            ]
        }
    
    Returns:
        {
            "success": true,
            "ranked": [
                {"answer": "Long-term control...", "score": 0.92, "rank": 1},
                {"answer": "Inhalers like...", "score": 0.78, "rank": 2},
                {"answer": "Try breathing...", "score": 0.45, "rank": 3}
            ]
        }
    """
    try:
        data = request.json
        prompt = data.get('prompt', '')
        candidates = data.get('candidates', [])
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": "'prompt' is required"
            }), 400
        
        if not candidates or len(candidates) < 2:
            return jsonify({
                "success": False,
                "error": "At least 2 candidate answers are required"
            }), 400
        
        from rlhf_reranker import rerank_candidates, is_model_ready
        
        if not is_model_ready():
            return jsonify({
                "success": False,
                "error": "Model not available. Please train the model first."
            }), 503
        
        ranked_raw = rerank_candidates(prompt, candidates)
        
        # Transform response to match frontend expectations
        ranked = []
        for idx, item in enumerate(ranked_raw, 1):
            ranked.append({
                "answer": item.get("text", ""),
                "score": item.get("_score", 0.0),
                "rank": idx
            })
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "ranked": ranked
        })
        
    except Exception as e:
        print(f"Error reranking answers: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================
# SFT Experiment Routes (pces_rlhf_experiments integration)
# ============================================

try:
    from sft_experiment_manager import (
        ensure_tables as sft_ensure_tables,
        get_ranked_data,
        add_ranked_entry,
        update_ranked_entry,
        delete_ranked_entry,
        delete_ranked_group,
        get_ranked_data_stats,
        import_from_jsonl,
        export_to_jsonl,
        list_experiments,
        get_experiment,
        delete_experiment,
        start_experiment,
        get_training_status,
        test_trained_model,
        get_department_list,
        get_prompts_by_department,
        recover_stuck_experiments,
        DEPARTMENTS,
        get_doctors,
        get_doctor_by_id,
        add_doctor,
        update_doctor,
        delete_doctor,
        get_doctors_by_departments,
        seed_sample_doctors,
        update_experiment_samples,
    )
    sft_ensure_tables()
    # Auto-import JSONL into local SQLite if empty
    try:
        import sft_experiment_manager as _sft_mod
        if _sft_mod._use_sqlite:
            stats = get_ranked_data_stats()
            if stats.get("success") and stats.get("total_entries", 0) == 0:
                jsonl_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "pces_rlhf_experiments", "medical_ranked.jsonl"
                )
                if os.path.exists(jsonl_path):
                    result = import_from_jsonl(jsonl_path)
                    print(f"📥 Auto-imported JSONL into SQLite: {result.get('imported', 0)} entries")
    except Exception as auto_err:
        print(f"⚠️ Auto-import skipped: {auto_err}")

    # Auto-recover stuck "running" experiments where model files exist
    try:
        recovered = recover_stuck_experiments()
        if recovered:
            print(f"🔄 Auto-recovered {recovered} stuck experiment(s)")
    except Exception as rec_err:
        print(f"⚠️ Recovery check skipped: {rec_err}")

    # Auto-seed sample doctors if table is empty
    try:
        seed_result = seed_sample_doctors()
        if seed_result.get("added", 0) > 0:
            print(f"👨‍⚕️ Auto-seeded {seed_result.get('added')} sample doctors")
    except Exception as seed_err:
        print(f"⚠️ Doctor seed skipped: {seed_err}")

    SFT_AVAILABLE = True
    print("✅ SFT Experiment Manager loaded")
except Exception as e:
    SFT_AVAILABLE = False
    print(f"⚠️ SFT Experiment Manager not available: {e}")


@app.route('/api/rlhf/experiments', methods=['GET'])
def api_list_experiments():
    """List all SFT experiments."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    return jsonify(list_experiments(page=page, per_page=per_page))


@app.route('/api/rlhf/experiment/create', methods=['POST'])
def api_create_experiment():
    """Create and start a new SFT training experiment."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    data = request.json or {}
    department = data.get('department')
    use_sme_scores = data.get('use_sme_scores', False)
    min_sme_score = data.get('min_sme_score', 1)
    
    # Generate name with score info if applicable
    score_label = f" (SME≥{min_sme_score})" if use_sme_scores else ""
    name = data.get('name') or (
        f"{department} SFT{score_label} {datetime.now().strftime('%Y-%m-%d %H:%M')}" if department
        else f"Experiment{score_label} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    config = {
        "model_name": data.get('model_name', 'microsoft/phi-2'),
        "lora_r": data.get('lora_r', 16),
        "lora_alpha": data.get('lora_alpha', 32),
        "lora_dropout": data.get('lora_dropout', 0.05),
        "num_epochs": data.get('num_epochs', 10),
        "batch_size": data.get('batch_size', 2),
        "gradient_accumulation_steps": data.get('gradient_accumulation_steps', 4),
        "learning_rate": data.get('learning_rate', 0.0001),
        "max_seq_length": data.get('max_seq_length', 2048),
    }
    result = start_experiment(name, config, department=department, use_sme_scores=use_sme_scores, min_sme_score=min_sme_score)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 400


@app.route('/api/rlhf/experiment/<int:exp_id>/status', methods=['GET'])
def api_experiment_status(exp_id):
    """Get real-time training status (for polling)."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    status = get_training_status()
    # Also include DB status for the experiment
    exp = get_experiment(exp_id)
    return jsonify({
        "success": True,
        "training": status,
        "experiment": exp.get("experiment") if exp.get("success") else None,
    })


@app.route('/api/rlhf/experiment/<int:exp_id>', methods=['GET'])
def api_get_experiment(exp_id):
    """Get experiment details."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_experiment(exp_id))


@app.route('/api/rlhf/experiment/<int:exp_id>', methods=['DELETE'])
def api_delete_experiment(exp_id):
    """Delete an experiment."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(delete_experiment(exp_id))


@app.route('/api/rlhf/experiment/<int:exp_id>/recalc-samples', methods=['POST'])
def api_recalc_experiment_samples(exp_id):
    """Recalculate and persist training_samples for an experiment using the live domain-aware count."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(update_experiment_samples(exp_id))


@app.route('/api/rlhf/experiment/<int:exp_id>/test', methods=['POST'])
def api_test_experiment_model(exp_id):
    """Test a trained model with a medical question."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    data = request.json or {}
    question = data.get('question', '')
    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400
    max_tokens = data.get('max_tokens', 256)
    result = test_trained_model(exp_id, question, max_new_tokens=max_tokens)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 400


@app.route('/api/rlhf/ranked-data', methods=['GET'])
def api_get_ranked_data():
    """Get ranked training data with optional filtering."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    group_id = request.args.get('group_id')
    search = request.args.get('search')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    sme_filter = request.args.get('sme_filter')           # None, 'reviewed', 'pending', 'high'
    domain = request.args.get('domain')                   # Department filter via domain column
    reason_empty = request.args.get('reason_empty') == '1'
    return jsonify(get_ranked_data(group_id=group_id, search=search, page=page, per_page=per_page, sme_filter=sme_filter, domain=domain, reason_empty=reason_empty))


@app.route('/api/rlhf/ranked-data', methods=['POST'])
def api_add_ranked_data():
    """Add a new ranked data group."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    data = request.json or {}
    prompt = data.get('prompt', '')
    responses = data.get('responses', [])
    if not prompt or not responses:
        return jsonify({"success": False, "error": "prompt and responses are required"}), 400
    return jsonify(add_ranked_entry(prompt, responses))


@app.route('/api/rlhf/ranked-data/<int:entry_id>', methods=['PUT'])
def api_update_ranked_data(entry_id):
    """Update a ranked data entry."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    data = request.json or {}
    return jsonify(update_ranked_entry(
        entry_id,
        response_text=data.get('response_text'),
        rank=data.get('rank'),
        reason=data.get('reason'),
    ))


@app.route('/api/rlhf/ranked-data/<int:entry_id>', methods=['DELETE'])
def api_delete_ranked_data(entry_id):
    """Delete a single ranked data entry."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(delete_ranked_entry(entry_id))


@app.route('/api/rlhf/ranked-data/group/<group_id>', methods=['DELETE'])
def api_delete_ranked_group(group_id):
    """Delete all entries in a ranked data group."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(delete_ranked_group(group_id))


@app.route('/api/rlhf/ranked-data/import', methods=['POST'])
def api_import_ranked_data():
    """Import ranked data from JSONL file."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    data = request.json or {}
    file_path = data.get('file_path')
    return jsonify(import_from_jsonl(file_path))


@app.route('/api/rlhf/ranked-data/export', methods=['GET'])
def api_export_ranked_data():
    """Export ranked data as JSONL."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    result = export_to_jsonl()
    if result.get("success"):
        from flask import Response
        return Response(
            result["data"],
            mimetype='application/jsonl',
            headers={'Content-Disposition': 'attachment; filename=medical_ranked_export.jsonl'}
        )
    return jsonify(result), 400


@app.route('/api/rlhf/ranked-data/stats', methods=['GET'])
def api_ranked_data_stats():
    """Get ranked data statistics."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_ranked_data_stats())


@app.route('/api/rlhf/departments', methods=['GET'])
def api_get_departments():
    """Get list of all medical departments."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_department_list())


@app.route('/api/rlhf/ranked-data/by-department', methods=['GET'])
def api_get_ranked_data_by_department():
    """Get prompts filtered by medical department and optionally by empty reason."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    department = request.args.get('department', '')
    limit = request.args.get('limit', 10, type=int)
    reason_empty_only = request.args.get('reason_empty', 'false').lower() == 'true'
    if not department:
        return jsonify({"success": False, "error": "department parameter is required"}), 400
    return jsonify(get_prompts_by_department(department, limit=limit, reason_empty_only=reason_empty_only))


# ============================================================
# DOCTOR MANAGEMENT API ROUTES
# ============================================================

@app.route('/api/rlhf/doctors', methods=['GET'])
def api_get_doctors():
    """Get list of doctors, optionally filtered by department."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    department = request.args.get('department', None)
    active_only = request.args.get('active_only', 'true').lower() == 'true'
    return jsonify(get_doctors(department=department, active_only=active_only))


@app.route('/api/rlhf/doctors/by-department', methods=['GET'])
def api_get_doctors_by_department():
    """Get all doctors grouped by department."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_doctors_by_departments())


@app.route('/api/rlhf/doctors/<int:doctor_id>', methods=['GET'])
def api_get_doctor(doctor_id):
    """Get a single doctor by ID."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    return jsonify(get_doctor_by_id(doctor_id))


@app.route('/api/rlhf/doctors', methods=['POST'])
def api_add_doctor():
    """Add a new doctor."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    data = request.json or {}
    name = data.get('name')
    department = data.get('department')
    email = data.get('email')
    specialty = data.get('specialty')
    
    if not name or not department:
        return jsonify({"success": False, "error": "name and department are required"}), 400
    
    result = add_doctor(name, department, email=email, specialty=specialty)
    if result.get("success"):
        return jsonify(result), 201
    return jsonify(result), 400


@app.route('/api/rlhf/doctors/<int:doctor_id>', methods=['PUT'])
def api_update_doctor(doctor_id):
    """Update an existing doctor."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    data = request.json or {}
    result = update_doctor(
        doctor_id,
        name=data.get('name'),
        email=data.get('email'),
        department=data.get('department'),
        specialty=data.get('specialty'),
        is_active=data.get('is_active')
    )
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 400


@app.route('/api/rlhf/doctors/<int:doctor_id>', methods=['DELETE'])
def api_delete_doctor(doctor_id):
    """Delete (deactivate) a doctor."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    hard_delete = request.args.get('hard', 'false').lower() == 'true'
    result = delete_doctor(doctor_id, hard_delete=hard_delete)
    if result.get("success"):
        return jsonify(result)
    return jsonify(result), 400


# ============================================================
# SME REVIEW API ROUTES
# ============================================================

@app.route('/api/rlhf/sme-review-queue', methods=['GET'])
def api_get_sme_review_queue():
    """Get prompts for SME review with filtering options."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    
    domain = request.args.get('domain', '')
    status = request.args.get('status', 'pending')  # 'pending', 'reviewed', 'all'
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    try:
        import sft_experiment_manager as sft
        with sft._connect() as conn:
            with conn.cursor() as cur:
                # Build query based on filters
                query = """
                    SELECT 
                        id,
                        prompt,
                        response_text,
                        rank,
                        reason,
                        group_id,
                        domain,
                        sme_score,
                        sme_score_reason,
                        sme_reviewed_by,
                        sme_reviewed_at,
                        created_by,
                        created_at,
                        updated_by,
                        updated_at
                    FROM sft_ranked_data
                    WHERE 1=1
                """
                params = []
                
                # Filter by domain
                if domain:
                    query += " AND LOWER(domain) = LOWER(%s)"
                    params.append(domain)
                
                # Filter by review status
                if status == 'pending':
                    query += " AND sme_score IS NULL"
                elif status == 'reviewed':
                    query += " AND sme_score IS NOT NULL"
                # 'all' means no additional filter
                
                # Add pagination
                query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                params.extend([per_page, (page - 1) * per_page])
                
                # Adapt SQL for SQLite if needed
                if sft._use_sqlite:
                    query = sft._adapt_sql(query)
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                # Get total count for pagination
                count_query = """
                    SELECT COUNT(*) FROM sft_ranked_data WHERE 1=1
                """
                count_params = []
                if domain:
                    count_query += " AND LOWER(domain) = LOWER(%s)"
                    count_params.append(domain)
                if status == 'pending':
                    count_query += " AND sme_score IS NULL"
                elif status == 'reviewed':
                    count_query += " AND sme_score IS NOT NULL"
                
                if sft._use_sqlite:
                    count_query = sft._adapt_sql(count_query)
                
                cur.execute(count_query, count_params)
                total = cur.fetchone()[0]
                
                items = []
                for row in rows:
                    items.append({
                        'id': row[0],
                        'prompt': row[1],
                        'response_text': row[2],
                        'rank': row[3],
                        'reason': row[4],
                        'group_id': row[5],
                        'domain': row[6],
                        'sme_score': row[7],
                        'sme_score_reason': row[8],
                        'sme_reviewed_by': row[9],
                        'sme_reviewed_at': sft._dt(row[10]),
                        'created_by': row[11],
                        'created_at': sft._dt(row[12]),
                        'updated_by': row[13],
                        'updated_at': sft._dt(row[14])
                    })
                
                return jsonify({
                    'success': True,
                    'items': items,
                    'total': total,
                    'page': page,
                    'per_page': per_page,
                    'total_pages': (total + per_page - 1) // per_page
                })
    
    except Exception as e:
        print(f"Error fetching SME review queue: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rlhf/sme-review-submit', methods=['POST'])
def api_submit_sme_reviews():
    """Batch update SME scores and reasons for multiple prompts."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    
    data = request.json or {}
    reviews = data.get('reviews', [])
    sme_name = data.get('sme_name', 'Unknown SME')
    
    if not reviews:
        return jsonify({'success': False, 'error': 'No reviews provided'}), 400
    
    try:
        import sft_experiment_manager as sft
        with sft._connect() as conn:
            with conn.cursor() as cur:
                updated_count = 0
                for review in reviews:
                    entry_id = review.get('id')
                    sme_score = review.get('sme_score')
                    sme_score_reason = review.get('sme_score_reason', '')
                    
                    if not entry_id or not sme_score:
                        continue
                    
                    # Validate score range
                    if sme_score < 1 or sme_score > 5:
                        continue
                    
                    query = """
                        UPDATE sft_ranked_data
                        SET sme_score = %s,
                            sme_score_reason = %s,
                            sme_reviewed_by = %s,
                            sme_reviewed_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """
                    if sft._use_sqlite:
                        query = sft._adapt_sql(query)
                    
                    cur.execute(query, (sme_score, sme_score_reason, sme_name, entry_id))
                    updated_count += cur.rowcount
                
                conn.commit()
                
                return jsonify({
                    'success': True,
                    'updated_count': updated_count,
                    'message': f'Successfully updated {updated_count} reviews'
                })
    
    except Exception as e:
        print(f"Error submitting SME reviews: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rlhf/sme-review-stats', methods=['GET'])
def api_get_sme_review_stats():
    """Get statistics about SME review progress."""
    if not SFT_AVAILABLE:
        return jsonify({"success": False, "error": "SFT module not available"}), 503
    
    domain = request.args.get('domain', '')
    
    try:
        import sft_experiment_manager as sft
        with sft._connect() as conn:
            with conn.cursor() as cur:
                # Overall stats
                stats_query = """
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN sme_score IS NOT NULL THEN 1 END) as reviewed,
                        COUNT(CASE WHEN sme_score IS NULL THEN 1 END) as pending
                    FROM sft_ranked_data
                """
                params = []
                if domain:
                    stats_query += " WHERE LOWER(domain) = LOWER(%s)"
                    params.append(domain)
                
                if sft._use_sqlite:
                    stats_query = sft._adapt_sql(stats_query)
                
                cur.execute(stats_query, params)
                row = cur.fetchone()
                total, reviewed, pending = row[0], row[1], row[2]
                
                # Reviews this week — use the correct date arithmetic for each backend
                if sft._use_sqlite:
                    week_query = """
                        SELECT COUNT(*)
                        FROM sft_ranked_data
                        WHERE sme_reviewed_at >= datetime('now', '-7 days')
                    """
                else:
                    week_query = """
                        SELECT COUNT(*)
                        FROM sft_ranked_data
                        WHERE sme_reviewed_at >= NOW() - INTERVAL '7 days'
                    """
                week_params = []
                if domain:
                    week_query += " AND LOWER(domain) = LOWER(%s)"
                    week_params.append(domain)
                
                if sft._use_sqlite:
                    week_query = sft._adapt_sql(week_query)
                
                cur.execute(week_query, week_params)
                this_week = cur.fetchone()[0]
                
                # Score distribution
                score_query = """
                    SELECT sme_score, COUNT(*) as count
                    FROM sft_ranked_data
                    WHERE sme_score IS NOT NULL
                """
                score_params = []
                if domain:
                    score_query += " AND LOWER(domain) = LOWER(%s)"
                    score_params.append(domain)
                score_query += " GROUP BY sme_score ORDER BY sme_score"
                
                if sft._use_sqlite:
                    score_query = sft._adapt_sql(score_query)
                
                cur.execute(score_query, score_params)
                score_dist = {row[0]: row[1] for row in cur.fetchall()}
                
                # Top reviewers
                reviewer_query = """
                    SELECT sme_reviewed_by, COUNT(*) as count
                    FROM sft_ranked_data
                    WHERE sme_reviewed_by IS NOT NULL
                """
                reviewer_params = []
                if domain:
                    reviewer_query += " AND LOWER(domain) = LOWER(%s)"
                    reviewer_params.append(domain)
                reviewer_query += " GROUP BY sme_reviewed_by ORDER BY count DESC LIMIT 5"
                
                if sft._use_sqlite:
                    reviewer_query = sft._adapt_sql(reviewer_query)
                
                cur.execute(reviewer_query, reviewer_params)
                top_reviewers = [{'name': row[0], 'count': row[1]} for row in cur.fetchall()]
                
                return jsonify({
                    'success': True,
                    'total': total,
                    'reviewed': reviewed,
                    'pending': pending,
                    'reviewed_this_week': this_week,
                    'review_percentage': round((reviewed / total * 100) if total > 0 else 0, 1),
                    'score_distribution': score_dist,
                    'top_reviewers': top_reviewers
                })
    
    except Exception as e:
        print(f"Error fetching SME review stats: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000, use_reloader=False)
