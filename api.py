# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import logging

from main import create_config, process_urls, process_url, compare_two, compare_products, save_products_to_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="AI Product Scraper API", version="1.1")

# Allow cross origin for UI convenience (you can tighten this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE_CONFIG = create_config()

@app.post("/scrape-urls")
def scrape_urls(body: Dict[str, Any]):
    """Body: { "urls": ["https://...","https://..."], "compare_all": true }"""
    try:
        urls = body.get("urls", [])
        compare_all = body.get("compare_all", False)
        if not urls or not isinstance(urls, list):
            raise HTTPException(status_code=400, detail="Provide a list of URLs in 'urls'.")
        products = process_urls(PIPELINE_CONFIG, urls, output_file="api_products.json", compare_all=False)
        comparisons = []
        if compare_all and len(products) >= 2:
            for i in range(len(products) - 1):
                for j in range(i + 1, len(products)):
                    comp_text = compare_products(products[i], products[j])
                    comparisons.append({
                        "pair": [products[i].get("title"), products[j].get("title")],
                        "comparison": comp_text
                    })
        return {
            "status": "success",
            "products": products,
            "comparisons": comparisons
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("scrape_urls failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-two")
def compare_two_endpoint(body: Dict[str, Any]):
    try:
        url1 = body.get("url1")
        url2 = body.get("url2")
        if not url1 or not url2:
            raise HTTPException(status_code=400, detail="url1 and url2 are required")
        result_text = compare_two(PIPELINE_CONFIG, url1, url2)
        if result_text is None:
            raise HTTPException(status_code=400, detail="Unable to extract from one or both URLs")
        return {"status": "success", "comparison": result_text}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("compare_two failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {
        "message": "Welcome to the AI Product Scraper API 🚀",
        "endpoints": {
            "/compare-two": "Compare two URLs"
        }
    }
