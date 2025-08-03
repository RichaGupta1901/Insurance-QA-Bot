from fastapi import FastAPI, Header, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import requests
from PyPDF2 import PdfReader
import io
import logging
from datetime import datetime
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Insurance Document Q&A API",
    description="Extract answers from insurance policy documents using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "meta-llama/llama-3.3-70b-instruct")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "12000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY not found in environment variables")
    raise ValueError("OPENROUTER_API_KEY is required")

class RequestBody(BaseModel):
    documents: HttpUrl  # For URL-based requests
    questions: List[str]
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if not v:
            raise ValueError("At least one question is required")
        if len(v) > 10:  # Reasonable limit
            raise ValueError("Maximum 10 questions allowed")
        for question in v:
            if not question.strip():
                raise ValueError("Questions cannot be empty")
        return v

class FileRequestBody(BaseModel):
    questions: List[str]
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if not v:
            raise ValueError("At least one question is required")
        if len(v) > 10:  # Reasonable limit
            raise ValueError("Maximum 10 questions allowed")
        for question in v:
            if not question.strip():
                raise ValueError("Questions cannot be empty")
        return v

class Answer(BaseModel):
    question: str
    answer: str
    confidence: Optional[str] = None
    processing_time_ms: Optional[int] = None

class Response(BaseModel):
    answers: List[Answer]
    document_info: Dict[str, Any]
    processing_summary: Dict[str, Any]

def extract_pdf_content(url: str) -> tuple[str, Dict[str, Any]]:
    """Extract text content from PDF URL with metadata"""
    try:
        logger.info(f"Downloading PDF from: {url}")
        
        # Download with timeout and proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        pdf_response = requests.get(
            str(url), 
            headers=headers, 
            timeout=REQUEST_TIMEOUT,
            stream=True
        )
        pdf_response.raise_for_status()
        
        # Check content type
        content_type = pdf_response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower():
            logger.warning(f"Unexpected content type: {content_type}")
        
        # Extract PDF content
        pdf_reader = PdfReader(io.BytesIO(pdf_response.content))
        
        # Extract text from all pages
        pages_text = []
        for i, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text() or ""
                pages_text.append(text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {i+1}: {e}")
                pages_text.append("")
        
        full_text = "\n".join(pages_text)
        
        # Document metadata
        doc_info = {
            "total_pages": len(pdf_reader.pages),
            "original_length": len(full_text),
            "content_hash": hashlib.md5(full_text.encode()).hexdigest()[:8],
            "extracted_at": datetime.utcnow().isoformat()
        }
        
        # Truncate if necessary
        if len(full_text) > MAX_TEXT_LENGTH:
            full_text = full_text[:MAX_TEXT_LENGTH]
            doc_info["truncated"] = True
            doc_info["truncated_length"] = MAX_TEXT_LENGTH
            logger.info(f"Text truncated to {MAX_TEXT_LENGTH} characters")
        else:
            doc_info["truncated"] = False
        
        return full_text, doc_info
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document parsing failed: {str(e)}"
        )
    """Extract text content from uploaded PDF file with metadata"""
    try:
        logger.info(f"Processing uploaded PDF: {filename}")
        
        # Extract PDF content
        pdf_reader = PdfReader(io.BytesIO(file_content))
        
        # Extract text from all pages
        pages_text = []
        for i, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text() or ""
                pages_text.append(text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {i+1}: {e}")
                pages_text.append("")
        
        full_text = "\n".join(pages_text)
        
        # Document metadata
        doc_info = {
            "filename": filename,
            "file_size_bytes": len(file_content),
            "total_pages": len(pdf_reader.pages),
            "original_length": len(full_text),
            "content_hash": hashlib.md5(full_text.encode()).hexdigest()[:8],
            "extracted_at": datetime.utcnow().isoformat()
        }
        
        # Truncate if necessary
        if len(full_text) > MAX_TEXT_LENGTH:
            full_text = full_text[:MAX_TEXT_LENGTH]
            doc_info["truncated"] = True
            doc_info["truncated_length"] = MAX_TEXT_LENGTH
            logger.info(f"Text truncated to {MAX_TEXT_LENGTH} characters")
        else:
            doc_info["truncated"] = False
        
        return full_text, doc_info
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PDF parsing failed: {str(e)}"
        )

def query_llm_with_retry(question: str, document_text: str) -> tuple[str, int]:
    """Query LLM with retry logic and timing"""
    start_time = time.time()
    
    prompt = f"""You are an expert assistant that analyzes insurance policy documents and provides accurate, concise answers.

DOCUMENT CONTENT:
{document_text}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the information provided in the document
- Be precise and concise
- If the information is not found in the document, clearly state "Information not found in the provided document"
- Include relevant policy numbers, dates, or specific terms when applicable
- For coverage amounts, deductibles, or limits, provide exact figures if available

ANSWER:"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app.com",  # Optional: helps with rate limiting
        "X-Title": "Insurance Document Q&A"
    }
    
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Lower for more consistent answers
        "max_tokens": 400,
        "top_p": 0.9
    }
    
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Querying LLM (attempt {attempt + 1}/{MAX_RETRIES})")
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check for API errors
            if "error" in result:
                raise Exception(f"API Error: {result['error']}")
            
            answer = result["choices"][0]["message"]["content"].strip()
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Successfully got LLM response in {processing_time}ms")
            return answer, processing_time
            
        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            logger.warning(f"Attempt {attempt + 1} timed out")
        except requests.exceptions.RequestException as e:
            last_error = f"Request failed: {str(e)}"
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            logger.error(f"Attempt {attempt + 1} error: {e}")
        
        # Wait before retry (exponential backoff)
        if attempt < MAX_RETRIES - 1:
            wait_time = 2 ** attempt
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    # All retries failed
    processing_time = int((time.time() - start_time) * 1000)
    return f"Error after {MAX_RETRIES} attempts: {last_error}", processing_time

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Insurance Document Q&A API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.head("/")
async def root_head():
    """Health check HEAD endpoint for Render"""
    return

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "model": MODEL,
        "max_text_length": MAX_TEXT_LENGTH,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/hackrx/run")
async def run_hackrx(request: RequestBody, authorization: Optional[str] = Header(None)):
    """
    Process insurance document from URL and answer questions - COMPETITION ENDPOINT
    
    - *documents*: Public URL to the PDF document
    - *questions*: List of questions to answer (max 10)
    """
    start_time = time.time()
    
    # Validate authorization header (if provided)
    if authorization and not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header. Use 'Bearer <token>'"
        )
    
    logger.info(f"Processing request with {len(request.questions)} questions")
    
    # Extract PDF content from URL
    document_text, doc_info = extract_pdf_content(request.documents)
    
    # Process each question
    answers = []
    total_llm_time = 0
    
    for i, question in enumerate(request.questions):
        logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
        
        answer_text, processing_time = query_llm_with_retry(question, document_text)
        total_llm_time += processing_time
        
        answers.append(answer_text)  # Competition expects simple string array
    
    total_time = int((time.time() - start_time) * 1000)
    
    logger.info(f"Request completed in {total_time}ms")
    
    # Return simple format as expected by competition
    return {"answers": answers}

@app.post("/hackrx/upload", response_model=Response)
async def run_hackrx_upload(
    file: UploadFile = File(...),
    questions: str = Form(...),
    authorization: Optional[str] = Header(None)
):
    """
    Process uploaded insurance document and answer questions
    
    - *file*: PDF file to upload
    - *questions*: JSON string array of questions (e.g., '["What is covered?", "What is the deductible?"]')
    """
    start_time = time.time()
    
    # Validate authorization header (if provided)
    if authorization and not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header. Use 'Bearer <token>'"
        )
    
    # Validate file type
    if not file.content_type or 'pdf' not in file.content_type.lower():
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
    
    # Parse questions from JSON string
    try:
        questions_list = json.loads(questions)
        if not isinstance(questions_list, list):
            raise ValueError("Questions must be a list")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid questions format. Use JSON array: {str(e)}"
        )
    
    # Validate questions using the same logic
    try:
        FileRequestBody(questions=questions_list)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    logger.info(f"Processing uploaded file: {file.filename} with {len(questions_list)} questions")
    
    # Read file content
    try:
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )
    
    # Extract PDF content
    document_text, doc_info = extract_pdf_content_from_file(file_content, file.filename)
    
    # Process each question
    answers = []
    total_llm_time = 0
    
    for i, question in enumerate(questions_list):
        logger.info(f"Processing question {i+1}/{len(questions_list)}: {question[:50]}...")
        
        answer_text, processing_time = query_llm_with_retry(question, document_text)
        total_llm_time += processing_time
        
        # Determine confidence based on response content
        confidence = "high"
        if "not found" in answer_text.lower() or "error" in answer_text.lower():
            confidence = "low"
        elif len(answer_text) < 20:
            confidence = "medium"
        
        answers.append(Answer(
            question=question,
            answer=answer_text,
            confidence=confidence,
            processing_time_ms=processing_time
        ))
    
    total_time = int((time.time() - start_time) * 1000)
    
    processing_summary = {
        "total_processing_time_ms": total_time,
        "llm_processing_time_ms": total_llm_time,
        "questions_processed": len(questions_list),
        "model_used": MODEL,
        "processed_at": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Upload request completed in {total_time}ms")
    
    return Response(
        answers=answers,
        document_info=doc_info,
        processing_summary=processing_summary
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An internal server error occurred"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # This must be 0.0.0.0 for Render
        port=port,
        reload=False,  # Disable reload for production
        log_level="info"
    )
