# main.py
import requests
from bs4 import BeautifulSoup
import re
import json
import time
import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from typing import Optional, Dict, List, Tuple
from textwrap import shorten
from pathlib import Path
# import ollama

from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

groq_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_KEY,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration (simple dict)
# ----------------------------
def create_config():
    return {
        "USER_AGENT": "MyScraperBot/1.0",
        "REQUEST_DELAY": 2,
        "REQUEST_TIMEOUT": 15,
        "MAX_TEXT_LENGTH": 12000,
        "DESCRIPTION_MAX_LENGTH": 300,
        "LLM_MODEL": "gemma3:1b",
        "LLM_TEXT_LIMIT": 3500,
        "USD_TO_INR": 89.0,
        "GBP_TO_INR": 104.5,
        "EUR_TO_INR": 102.0,
        "PRICE_PATTERN": r"(₹\s?\d[\d,]*(?:\.\d+)?|\$\s?\d[\d,]*(?:\.\d+)?|€\s?\d[\d,]*(?:\.\d+)?|£\s?\d[\d,]*(?:\.\d+)?)",
        "RATING_PATTERN": r"(\b\d(?:\.\d)?\s?(?:\/\s?5|stars?)\b|\b\d(?:\.\d)?(?=\s?\/\s?5))"
    }

# ----------------------------
# Robots check (cached)
# ----------------------------
_ROBOTS_CACHE: Dict[str, bool] = {}

def is_allowed_by_robots(url: str, user_agent: str) -> bool:
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.warning("Invalid URL for robots check: %s", url)
            return False
        cache_key = f"{parsed.netloc}:{user_agent}"
        if cache_key in _ROBOTS_CACHE:
            return _ROBOTS_CACHE[cache_key]
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception as e:
            logger.warning("Could not read robots.txt (%s): %s - allowing by default", robots_url, e)
            _ROBOTS_CACHE[cache_key] = True
            return True
        allowed = rp.can_fetch(user_agent, url)
        _ROBOTS_CACHE[cache_key] = allowed
        logger.info("Scraping %s allowed=%s", url, allowed)
        return allowed
    except Exception as e:
        logger.error("Robots check error for %s: %s", url, e)
        return False

# ----------------------------
# Fetch & extract
# ----------------------------
_session: Optional[requests.Session] = None

def _get_session(config):
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": config["USER_AGENT"]})
    return _session

def fetch_html(config, url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.warning("Invalid URL: %s", url)
            return None
        if not is_allowed_by_robots(url, config["USER_AGENT"]):
            logger.warning("Blocked by robots.txt: %s", url)
            return None
        sess = _get_session(config)
        resp = sess.get(url, timeout=config["REQUEST_TIMEOUT"])
        resp.raise_for_status()
        time.sleep(config["REQUEST_DELAY"])
        logger.info("Fetched HTML: %s", url)
        return resp.text
    except requests.exceptions.Timeout:
        logger.error("Timeout fetching: %s", url)
    except requests.exceptions.RequestException as e:
        logger.error("Request error: %s -> %s", url, e)
    except Exception as e:
        logger.error("Unexpected fetch error: %s -> %s", url, e)
    return None

def extract_page_data(config, url: str) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        html = fetch_html(config, url)
        if not html:
            return None, "Failed to fetch or blocked by robots.txt"
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        extracted_text = soup.get_text(separator=" ", strip=True)
        if not extracted_text:
            return None, "No text extracted from page"
        prices = list(dict.fromkeys(re.findall(config["PRICE_PATTERN"], extracted_text)))
        price_candidates = prices[:]
        price_tag = soup.find(lambda tag: tag.name in ("p", "span") and re.search(r'price|amount|cost', ' '.join(tag.get('class', [])), re.I))
        if price_tag:
            txt = price_tag.get_text(strip=True)
            if txt and txt not in price_candidates:
                price_candidates.insert(0, txt)
        ratings = []
        star_elems = soup.find_all(class_=re.compile(r'star[-_ ]?rating|star|rating', re.I))
        rating_class_map = {'one': '1/5', 'two': '2/5', 'three': '3/5', 'four': '4/5', 'five': '5/5'}
        for elem in star_elems:
            classes = elem.get("class", []) or []
            classes_text = " ".join(classes)
            m = re.search(r'\b(one|two|three|four|five)\b', classes_text, re.I)
            if m:
                ratings.append(rating_class_map[m.group(1).lower()])
                continue
            txt = elem.get_text(" ", strip=True)
            if txt:
                ratings.append(txt)
        if not ratings:
            textual_matches = list(dict.fromkeys(re.findall(config["RATING_PATTERN"], extracted_text)))
            ratings.extend(textual_matches)
        seen = set()
        ratings = [r for r in ratings if not (r in seen or seen.add(r))]
        sentences = [s.strip() for s in re.split(r'[.!?\n]', extracted_text) if len(s.strip()) > 20]
        description = max(sentences, key=len, default="")[:config["DESCRIPTION_MAX_LENGTH"]]
        logger.info("Extracted page data: %s", url)
        return {
            "text": extracted_text[:config["MAX_TEXT_LENGTH"]],
            "prices": price_candidates,
            "rating_candidates": ratings,
            "description_snippet": description.strip()
        }, None
    except Exception as e:
        logger.exception("Error extracting page data from %s: %s", url, e)
        return None, f"Unexpected extraction error: {e}"

# ----------------------------
# Currency helpers (functions)
# ----------------------------
def parse_amount(text: str) -> float:
    if not text:
        return 0.0
    cleaned = text.replace(",", "").strip()
    match = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    return float(match.group(1)) if match else 0.0

def to_inr_float(config, value: str) -> float:
    if not value:
        return 0.0
    value = value.strip()
    try:
        if "₹" in value:
            return parse_amount(value)
        if "$" in value:
            return round(parse_amount(value) * config["USD_TO_INR"], 2)
        if "£" in value:
            return round(parse_amount(value) * config["GBP_TO_INR"], 2)
        if "€" in value:
            return round(parse_amount(value) * config["EUR_TO_INR"], 2)
        if re.match(r'^[\d,\.]+$', value):
            return round(parse_amount(value) * config["USD_TO_INR"], 2)
    except Exception as e:
        logger.warning("Currency conversion failed for '%s': %s", value, e)
    return 0.0

def format_inr(amount: float) -> str:
    if amount is None or amount == 0.0:
        return ""
    return f"₹{amount:,.2f}"

# ----------------------------
# Rating normalize
# ----------------------------
_WORD_MAP = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}

def normalize_rating(raw: str) -> Optional[float]:
    if not raw:
        return None
    s = str(raw).lower().strip()
    frac = re.search(r'(\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)', s)
    if frac:
        try:
            val = float(frac.group(1))
            maxv = float(frac.group(2)) if float(frac.group(2)) > 0 else 5.0
            if maxv != 5.0:
                val = (val / maxv) * 5.0
            return max(0.0, min(5.0, round(val, 2)))
        except:
            pass
    s = s.replace("stars", "").replace("star", "").replace("out of 5", "").replace("/5", "").strip()
    if s in _WORD_MAP:
        return float(_WORD_MAP[s])
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(5.0, val))
        except:
            pass
    return None

# ----------------------------
# LLM extraction (function)
# ----------------------------
def _clean_json_response(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON payload")
    return {}

def extract_product_with_llm(config, url: str, raw_data: Dict) -> Dict:
    page_text = raw_data.get("text", "")
    prices = raw_data.get("prices", []) or []
    ratings = raw_data.get("rating_candidates", []) or []
    description = raw_data.get("description_snippet", "")

    safe_text = shorten(page_text, width=config["LLM_TEXT_LIMIT"], placeholder="... [truncated]")

    prompt_data = {
        "description_snippet": description,
        "price_options": prices[:5],
        "rating_options": ratings[:3]
    }

    prompt = f"""Extract product information from this webpage data.

INPUT DATA:
URL: {url}

Available Data:
{json.dumps(prompt_data, ensure_ascii=False, indent=2)}

PAGE CONTEXT:
{safe_text}

INSTRUCTIONS:
Return ONLY valid JSON with these exact fields:
- "title": Product name/title
- "price": Best matching price from price_options (select the main product price) — or empty string if not found
- "description": Clear 2-3 sentence product description
- "rating": Numeric rating or word (e.g., "Four")
- "link": Use provided URL exactly

RULES:
1. Output ONLY JSON, no explanations
2. Empty string "" if field unavailable

JSON OUTPUT:
{{
  "title": "",
  "price": "",
  "description": "",
  "rating": "",
  "link": "{url}"
}}"""

    # try:
    #     response = ollama.chat(model=config["LLM_MODEL"], messages=[{"role": "user", "content": prompt}])
    #     raw_output = response["message"]["content"]
    #     data = _clean_json_response(raw_output)
    # except Exception as e:
    #     logger.warning("LLM call failed: %s", e)
    #     data = {}

    try:
        response = groq_model.invoke(prompt)
        raw_output = response.content
        data = _clean_json_response(raw_output)

    except Exception as e:
        logger.warning("Groq LLM call failed: %s", e)
        data = {}


    title = (data.get("title") or "").strip()
    raw_price = (data.get("price") or "").strip()
    raw_rating = data.get("rating") or ""

    price_inr_val = 0.0
    if raw_price:
        price_inr_val = to_inr_float(config, raw_price)
    if price_inr_val == 0.0 and prices:
        price_inr_val = to_inr_float(config, prices[0])
    price_inr_str = format_inr(price_inr_val)

    rating_val = normalize_rating(raw_rating)
    if rating_val is None and ratings:
        rating_val = normalize_rating(ratings[0])

    return {
        "title": title,
        "price": price_inr_str,
        "price_inr": price_inr_val,
        "description": (data.get("description") or description or "").strip(),
        "rating": float(rating_val) if rating_val is not None else None,
        "link": url
    }

# ----------------------------
# Analyzer: summarise & compare
# ----------------------------
def summarize_product(product: Dict) -> str:
    if not product or product.get("error"):
        return "❌ Unable to generate summary - invalid product data"
    title = product.get("title") or "Not available"
    price = product.get("price") or "Not available"
    rating = f"{product.get('rating')}/5" if product.get('rating') is not None else "Not available"
    description = product.get("description") or "Not available"
    bullets = []
    for s in re.split(r'[\.\n]', description):
        s = s.strip()
        if s and len(bullets) < 3 and len(s) < 80:
            bullets.append(s)
    if not bullets:
        bullets = ["Not available"]
    out = [f"Title: {title}", f"Price: {price}", f"Rating: {rating}", "", "Key Features:"]
    for b in bullets:
        out.append(f"• {b}")
    return "\n".join(out)

def compare_products(p1: Dict, p2: Dict) -> str:
    if not p1 or not p2:
        return "❌ Cannot compare - invalid product data"
    title1 = p1.get("title") or "Product 1"
    title2 = p2.get("title") or "Product 2"
    price1_val = p1.get("price_inr") or 0.0
    price2_val = p2.get("price_inr") or 0.0
    price1_str = p1.get("price") or "Not available"
    price2_str = p2.get("price") or "Not available"
    if price1_val and price2_val:
        if price1_val < price2_val:
            pricing_winner = title1
        elif price2_val < price1_val:
            pricing_winner = title2
        else:
            pricing_winner = "Equal prices"
    elif price1_val and not price2_val:
        pricing_winner = title1
    elif price2_val and not price1_val:
        pricing_winner = title2
    else:
        pricing_winner = "Not available"
    r1 = p1.get("rating")
    r2 = p2.get("rating")
    if r1 is not None and r2 is not None:
        if r1 > r2:
            rating_winner = title1
        elif r2 > r1:
            rating_winner = title2
        else:
            rating_winner = "Equal ratings"
    elif r1 is not None:
        rating_winner = title1
    elif r2 is not None:
        rating_winner = title2
    else:
        rating_winner = "Not available"
    def extract_features(prod):
        desc = prod.get("description") or ""
        features = []
        for s in re.split(r'[.\n]', desc):
            s = s.strip()
            if s and len(features) < 3:
                features.append(s if len(s) < 80 else s[:77] + "...")
        return features or ["Not available"]
    f1 = extract_features(p1)
    f2 = extract_features(p2)
    recommendation = "Not available"
    if rating_winner != "Not available" and rating_winner != "Equal ratings":
        recommendation = f"Recommend {rating_winner} because it has a higher rating."
    elif rating_winner == "Equal ratings" and pricing_winner not in ("Not available", "Equal prices"):
        recommendation = f"Recommend {pricing_winner} because ratings are equal and it is cheaper."
    elif pricing_winner not in ("Not available", "Equal prices"):
        recommendation = f"Recommend {pricing_winner} based on price (cheaper)."
    else:
        recommendation = "No clear recommendation based on provided data."
    cmp_lines = [
        "📊 PRODUCT COMPARISON",
        "",
        "1. PRICING",
        f"   • {title1}: {price1_str}",
        f"   • {title2}: {price2_str}",
        f"   → Better Value: {pricing_winner}",
        "",
        "2. RATINGS",
        f"   • {title1}: {r1 if r1 is not None else 'Not available'}/5",
        f"   • {title2}: {r2 if r2 is not None else 'Not available'}/5",
        f"   → Better Rated: {rating_winner}",
        "",
        "3. FEATURES",
        f"   {title1}:",
    ]
    for item in f1:
        cmp_lines.append(f"   • {item}")
    cmp_lines.append("")
    cmp_lines.append(f"   {title2}:")
    for item in f2:
        cmp_lines.append(f"   • {item}")
    cmp_lines.append("")
    cmp_lines.append("4. RECOMMENDATION")
    cmp_lines.append(f"   {recommendation}")
    return "\n".join(cmp_lines)

# ----------------------------
# Product save/load
# ----------------------------
def deduplicate_products(products: List[Dict]) -> List[Dict]:
    unique = []
    seen = set()
    for p in products:
        link = p.get("link", "")
        if link and link not in seen:
            unique.append(p)
            seen.add(link)
    return unique

def save_products_to_json(products: List[Dict], filename: str = "products.json") -> None:
    filepath = Path(filename)
    unique = deduplicate_products(products)
    valid_products = [p for p in unique if not p.get("error")]
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(valid_products, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d products to %s", len(valid_products), filename)
        print(f"\n✔ Successfully saved {len(valid_products)} products → {filename}")
    except Exception as e:
        logger.error("Error saving JSON: %s", e)
        print(f"❌ Error saving JSON: {e}")

def load_products_from_json(filename: str = "products.json") -> List[Dict]:
    filepath = Path(filename)
    if not filepath.exists():
        logger.warning("File not found: %s", filename)
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading JSON: %s", e)
        return []

# ----------------------------
# Pipeline functions (public)
# ----------------------------
def process_url(config, url: str, show_summary: bool = True) -> Optional[Dict]:
    print(f"\n🔍 Processing: {url}")
    logger.info("Starting to process URL: %s", url)
    raw_data, error = extract_page_data(config, url)
    if error:
        print(f"❌ Error: {error}")
        return None
    print("✔ Extracting with AI...")
    product = extract_product_with_llm(config, url, raw_data)
    if product.get("error"):
        print(f"❌ Extraction error: {product['error']}")
        return None
    print("\n📦 Extracted Product:")
    print(f"   Title: {product.get('title', 'N/A')}")
    print(f"   Price: {product.get('price', 'N/A')}")
    print(f"   Rating: {product.get('rating', 'N/A')}/5")
    if show_summary:
        print("\n📝 AI Summary:")
        print("─" * 60)
        summary = summarize_product(product)
        print(summary)
        print("─" * 60)
    return product

def process_urls(config, urls: List[str], output_file: str = "products.json", compare_all: bool = False) -> List[Dict]:
    print("\n" + "=" * 60)
    print("PRODUCT SCRAPER PIPELINE STARTED")
    print("=" * 60)
    products = []
    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}]")
        product = process_url(config, url, show_summary=True)
        if product and not product.get("error"):
            products.append(product)
    if products:
        save_products_to_json(products, output_file)
    if compare_all and len(products) >= 2:
        print("\n" + "=" * 60)
        print("PRODUCT COMPARISONS")
        print("=" * 60)
        for i in range(len(products) - 1):
            for j in range(i + 1, len(products)):
                print(f"\n🔄 Comparing Product {i+1} vs Product {j+1}:")
                print("─" * 60)
                comparison = compare_products(products[i], products[j])
                print(comparison)
                print("─" * 60)
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE - Processed {len(products)} products")
    print("=" * 60)
    return products

def compare_two(config, url1: str, url2: str) -> Optional[str]:
    p1 = process_url(config, url1, show_summary=False)
    p2 = process_url(config, url2, show_summary=False)
    products = []
    if p1:
        products.append(p1)
    if p2:
        products.append(p2)
    if len(products) == 2:
        save_products_to_json(products)
        return compare_products(products[0], products[1])
    return None

def cleanup():
    global _session
    if _session:
        try:
            _session.close()
        except Exception:
            pass
    _session = None

# ----------------------------
# Simple CLI usage
# ----------------------------
def main():
    cfg = create_config()
    urls = [
        "https://www.flipkart.com/deals4you-sneakers-women/p/itmb224093aba791?pid=SHOGD2FQ9HEUD2JC&lid=LSTSHOGD2FQ9HEUD2JCR4TJLZ&marketplace=FLIPKART",
        "https://www.flipkart.com/red-tape-women-s-athleisure-sports-shoes-active-everyday-style-walking-women/p/itm5b2f4a85a0cf5?pid=SHOHFKMBKQBYWGZ3&lid=LSTSHOHFKMBKQBYWGZ3KWB8GG&marketplace=FLIPKART"
    ]
    try:
        process_urls(cfg, urls, output_file="url_data.json", compare_all=True)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
