# api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime
from pathlib import Path
import json
import uuid

from main1 import (
    create_config, 
    process_urls, 
    process_url, 
    compare_two, 
    compare_products, 
    save_products_to_json,
    load_products_from_json,
    cleanup
)

# ====================================
# Logging Configuration
# ====================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# ====================================
# FastAPI App Initialization
# ====================================
app = FastAPI(
    title="AI Product Scraper API",
    description="Advanced product comparison API with AI-powered extraction",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ====================================
# CORS Middleware
# ====================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================
# Global Configuration
# ====================================
PIPELINE_CONFIG = create_config()
STORAGE_DIR = Path("data")
STORAGE_DIR.mkdir(exist_ok=True)

# Task tracking for async operations
TASK_STORAGE: Dict[str, Dict[str, Any]] = {}

# ====================================
# Pydantic Models
# ====================================
class URLInput(BaseModel):
    url: str = Field(..., description="Product URL to scrape", min_length=10)
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class ScrapeURLsRequest(BaseModel):
    urls: List[str] = Field(..., description="List of product URLs", min_items=1, max_items=10)
    compare_all: bool = Field(default=False, description="Compare all products")
    use_selenium: Optional[bool] = Field(default=None, description="Force Selenium usage")
    save_results: bool = Field(default=True, description="Save results to file")
    
    @validator('urls')
    def validate_urls(cls, v):
        if not v:
            raise ValueError('URLs list cannot be empty')
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f'Invalid URL: {url}')
        return v

class CompareTwoRequest(BaseModel):
    url1: str = Field(..., description="First product URL")
    url2: str = Field(..., description="Second product URL")
    use_selenium: Optional[bool] = Field(default=None, description="Force Selenium usage")
    
    @validator('url1', 'url2')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class ProductResponse(BaseModel):
    title: str
    price: str
    price_inr: float
    description: str
    rating: Optional[float]
    link: str

class ComparisonResponse(BaseModel):
    pair: List[str]
    comparison: str

class ScrapeResponse(BaseModel):
    status: str
    message: str
    products: List[Dict[str, Any]]
    comparisons: List[Dict[str, Any]]
    execution_time: float
    timestamp: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# ====================================
# Middleware for Request Logging
# ====================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Request completed in {process_time:.2f}s - Status: {response.status_code}")
    
    return response

# ====================================
# Exception Handlers
# ====================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error occurred",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ====================================
# Helper Functions
# ====================================
def generate_task_id() -> str:
    """Generate unique task ID"""
    return str(uuid.uuid4())

def save_task_result(task_id: str, result: Dict[str, Any]):
    """Save task result to storage"""
    try:
        filepath = STORAGE_DIR / f"task_{task_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Task result saved: {task_id}")
    except Exception as e:
        logger.error(f"Failed to save task result: {e}")

def load_task_result(task_id: str) -> Optional[Dict[str, Any]]:
    """Load task result from storage"""
    try:
        filepath = STORAGE_DIR / f"task_{task_id}.json"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load task result: {e}")
    return None

async def process_urls_async(task_id: str, urls: List[str], config: Dict, compare_all: bool):
    """Asynchronous URL processing"""
    try:
        TASK_STORAGE[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting extraction..."
        }
        
        # Process URLs
        products = []
        for idx, url in enumerate(urls):
            TASK_STORAGE[task_id]["progress"] = int((idx / len(urls)) * 80)
            TASK_STORAGE[task_id]["message"] = f"Processing URL {idx + 1}/{len(urls)}"
            
            product = process_url(config, url, show_summary=False)
            if product and not product.get("error"):
                products.append(product)
        
        # Generate comparisons
        comparisons = []
        if compare_all and len(products) >= 2:
            TASK_STORAGE[task_id]["progress"] = 85
            TASK_STORAGE[task_id]["message"] = "Generating comparisons..."
            
            for i in range(len(products) - 1):
                for j in range(i + 1, len(products)):
                    comp_text = compare_products(products[i], products[j])
                    comparisons.append({
                        "pair": [products[i].get("title"), products[j].get("title")],
                        "comparison": comp_text
                    })
        
        # Save results
        result = {
            "status": "completed",
            "products": products,
            "comparisons": comparisons,
            "timestamp": datetime.now().isoformat()
        }
        
        save_task_result(task_id, result)
        
        TASK_STORAGE[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Processing complete",
            "result": result
        }
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        TASK_STORAGE[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": str(e)
        }

# ====================================
# API Endpoints
# ====================================

@app.get("/", tags=["General"])
async def root():
    """API Root - Welcome message and available endpoints"""
    return {
        "message": "🚀 AI Product Scraper API v2.0",
        "description": "Advanced product comparison with AI extraction",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /scrape-urls": "Scrape and compare multiple URLs",
            "POST /scrape-urls/async": "Async scraping with task tracking",
            "POST /compare-two": "Compare two specific URLs",
            "POST /scrape-single": "Scrape single URL",
            "GET /task/{task_id}": "Check async task status",
            "GET /products": "List saved products",
            "GET /config": "Get current configuration"
        },
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/config", tags=["Configuration"])
async def get_config():
    """Get current API configuration"""
    safe_config = PIPELINE_CONFIG.copy()
    # Remove sensitive data if any
    return {
        "status": "success",
        "config": safe_config,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/scrape-urls", response_model=ScrapeResponse, tags=["Scraping"])
async def scrape_urls(request: ScrapeURLsRequest):
    """
    Scrape multiple product URLs and optionally compare them
    
    - **urls**: List of product URLs to scrape (1-10 URLs)
    - **compare_all**: If true, compare all products pairwise
    - **use_selenium**: Force Selenium usage (overrides config)
    - **save_results**: Save results to JSON file
    """
    start_time = datetime.now()
    
    try:
        # Configure Selenium if specified
        config = PIPELINE_CONFIG.copy()
        if request.use_selenium is not None:
            config["USE_SELENIUM"] = request.use_selenium
        
        logger.info(f"Starting to scrape {len(request.urls)} URLs")
        
        # Process URLs
        output_file = f"api_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json" if request.save_results else None
        products = process_urls(
            config, 
            request.urls, 
            output_file=output_file or "temp_products.json",
            compare_all=False
        )
        
        # Generate comparisons if requested
        comparisons = []
        if request.compare_all and len(products) >= 2:
            logger.info("Generating product comparisons")
            for i in range(len(products) - 1):
                for j in range(i + 1, len(products)):
                    comp_text = compare_products(products[i], products[j])
                    comparisons.append({
                        "pair": [products[i].get("title", f"Product {i+1}"), 
                                products[j].get("title", f"Product {j+1}")],
                        "comparison": comp_text
                    })
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Successfully processed {len(products)} products in {execution_time:.2f}s")
        
        return ScrapeResponse(
            status="success",
            message=f"Successfully processed {len(products)} products",
            products=products,
            comparisons=comparisons,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.exception("scrape_urls failed")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@app.post("/scrape-urls/async", tags=["Scraping"])
async def scrape_urls_async_endpoint(
    request: ScrapeURLsRequest,
    background_tasks: BackgroundTasks
):
    """
    Asynchronously scrape multiple URLs and track progress
    
    Returns a task_id that can be used to check status via /task/{task_id}
    """
    try:
        task_id = generate_task_id()
        
        # Configure Selenium if specified
        config = PIPELINE_CONFIG.copy()
        if request.use_selenium is not None:
            config["USE_SELENIUM"] = request.use_selenium
        
        # Add task to background
        background_tasks.add_task(
            process_urls_async,
            task_id,
            request.urls,
            config,
            request.compare_all
        )
        
        logger.info(f"Created async task: {task_id}")
        
        return {
            "status": "accepted",
            "message": "Task created successfully",
            "task_id": task_id,
            "check_status_url": f"/task/{task_id}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("Failed to create async task")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(task_id: str):
    """
    Get status of an async scraping task
    
    - **task_id**: Task ID returned from async scraping endpoint
    """
    # Check in-memory storage first
    if task_id in TASK_STORAGE:
        task_data = TASK_STORAGE[task_id]
        return TaskStatusResponse(
            task_id=task_id,
            status=task_data.get("status", "unknown"),
            progress=task_data.get("progress", 0),
            message=task_data.get("message"),
            result=task_data.get("result")
        )
    
    # Check file storage
    result = load_task_result(task_id)
    if result:
        return TaskStatusResponse(
            task_id=task_id,
            status="completed",
            progress=100,
            message="Task completed (retrieved from storage)",
            result=result
        )
    
    raise HTTPException(status_code=404, detail="Task not found")

@app.post("/scrape-single", tags=["Scraping"])
async def scrape_single_url(request: URLInput):
    """
    Scrape a single product URL
    
    - **url**: Product URL to scrape
    """
    try:
        logger.info(f"Scraping single URL: {request.url}")
        
        product = process_url(PIPELINE_CONFIG, request.url, show_summary=False)
        
        if product and not product.get("error"):
            return {
                "status": "success",
                "product": product,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail="Failed to extract product data"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("scrape_single failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-two", tags=["Comparison"])
async def compare_two_endpoint(request: CompareTwoRequest):
    """
    Compare two specific product URLs
    
    - **url1**: First product URL
    - **url2**: Second product URL
    - **use_selenium**: Force Selenium usage
    """
    try:
        logger.info(f"Comparing URLs: {request.url1} vs {request.url2}")
        
        # Configure Selenium if specified
        config = PIPELINE_CONFIG.copy()
        if request.use_selenium is not None:
            config["USE_SELENIUM"] = request.use_selenium
        
        result_text = compare_two(config, request.url1, request.url2)
        
        if result_text is None:
            raise HTTPException(
                status_code=400, 
                detail="Unable to extract from one or both URLs"
            )
        
        # Load saved products for detailed info
        products = load_products_from_json("products.json")
        
        return {
            "status": "success",
            "comparison": result_text,
            "products": products[-2:] if len(products) >= 2 else products,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("compare_two failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products", tags=["Products"])
async def list_products(limit: int = 50, offset: int = 0):
    """
    List saved products with pagination
    
    - **limit**: Maximum number of products to return (default: 50)
    - **offset**: Number of products to skip (default: 0)
    """
    try:
        products = load_products_from_json("products.json")
        
        total = len(products)
        paginated = products[offset:offset + limit]
        
        return {
            "status": "success",
            "products": paginated,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "returned": len(paginated)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("list_products failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/download", tags=["Products"])
async def download_products():
    """
    Download all saved products as JSON file
    """
    try:
        filepath = Path("products.json")
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="No products found")
        
        return FileResponse(
            path=filepath,
            filename=f"products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            media_type="application/json"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("download_products failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/products", tags=["Products"])
async def clear_products():
    """
    Clear all saved products
    """
    try:
        filepath = Path("products.json")
        if filepath.exists():
            filepath.unlink()
        
        return {
            "status": "success",
            "message": "Products cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("clear_products failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup", tags=["Maintenance"])
async def cleanup_resources():
    """
    Cleanup resources (Selenium drivers, sessions, etc.)
    """
    try:
        cleanup()
        
        return {
            "status": "success",
            "message": "Resources cleaned up successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("cleanup failed")
        raise HTTPException(status_code=500, detail=str(e))

# ====================================
# Startup & Shutdown Events
# ====================================
@app.on_event("startup")
async def startup_event():
    """Execute on API startup"""
    logger.info("=" * 60)
    logger.info("AI Product Scraper API Starting Up")
    logger.info("=" * 60)
    logger.info(f"Storage directory: {STORAGE_DIR.absolute()}")
    logger.info(f"Selenium enabled: {PIPELINE_CONFIG.get('USE_SELENIUM', False)}")
    logger.info("API is ready to accept requests")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Execute on API shutdown"""
    logger.info("=" * 60)
    logger.info("AI Product Scraper API Shutting Down")
    logger.info("=" * 60)
    
    # Cleanup resources
    try:
        cleanup()
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
    
    logger.info("Shutdown complete")
    logger.info("=" * 60)

# ====================================
# Run Server (for development)
# ====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
