import os
import json
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
import time
import re
from typing import List, Dict, Optional

# ====================================
# Configuration
# ====================================
load_dotenv()
#API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = "https://ai-shopping-helper-1.onrender.com"

st.set_page_config(
    page_title="🛒 AI Shopping Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================
# Custom CSS
# ====================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .product-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .price-tag {
        font-size: 1.8rem;
        font-weight: bold;
        color: #28a745;
    }
    .rating-badge {
        background-color: #ffc107;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .comparison-box {
        background-color: #e7f3ff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #1f77b4;
    }
    .winner-badge {
        background-color: #28a745;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .error-box {
        background-color: #ffe6e6;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e6ffe6;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
</style>
""", unsafe_allow_html=True)

# ====================================
# Session State Initialization
# ====================================
if "processed" not in st.session_state:
    st.session_state.processed = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "saved_file" not in st.session_state:
    st.session_state.saved_file = ""
if "task_id" not in st.session_state:
    st.session_state.task_id = None
if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = "sync"
if "url_count" not in st.session_state:
    st.session_state.url_count = 2
if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = []

# ====================================
# Helper Functions
# ====================================
def validate_url(url: str) -> bool:
    """Validate URL format"""
    pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(pattern.match(url))

def extract_features(text: str, max_items: int = 3) -> List[str]:
    """Extract key features from description"""
    if not text:
        return ["Not available"]
    
    pieces = [s.strip() for s in text.split('.') if s.strip()]
    features = []
    for p in pieces:
        if len(features) >= max_items:
            break
        if len(p) > 8:
            features.append(p if len(p) < 120 else p[:117] + '...')
    
    return features or ["Not available"]

def format_price(price_str: str) -> str:
    """Format price for display"""
    if not price_str or price_str == "Not available":
        return "Price not available"
    return price_str

def get_best_product(products: List[Dict]) -> Optional[Dict]:
    """Determine best product based on rating and price"""
    if not products:
        return None
    
    valid_products = [p for p in products if p.get("rating") is not None and p.get("price_inr", 0) > 0]
    
    if not valid_products:
        return products[0] if products else None
    
    # Score based on rating (70%) and inverse price (30%)
    for p in valid_products:
        rating_score = (p.get("rating", 0) / 5.0) * 0.7
        max_price = max(p.get("price_inr", 1) for p in valid_products)
        price_score = (1 - (p.get("price_inr", max_price) / max_price)) * 0.3
        p["_score"] = rating_score + price_score
    
    return max(valid_products, key=lambda x: x.get("_score", 0))

def check_api_health() -> bool:
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def poll_task_status(task_id: str, max_attempts: int = 60) -> Optional[Dict]:
    """Poll async task status until completion"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_URL}/task/{task_id}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                progress = data.get("progress", 0)
                message = data.get("message", "Processing...")
                
                progress_bar.progress(progress / 100)
                status_text.text(f"Status: {status} - {message}")
                
                if status == "completed":
                    progress_bar.empty()
                    status_text.empty()
                    return data.get("result")
                elif status == "failed":
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Task failed: {message}")
                    return None
                
                time.sleep(2)
            else:
                time.sleep(2)
        except Exception as e:
            st.error(f"Error checking task status: {e}")
            time.sleep(2)
    
    progress_bar.empty()
    status_text.empty()
    st.warning("Task status check timed out")
    return None

# ====================================
# Visualization Functions
# ====================================
def create_price_comparison_chart(products: List[Dict]) -> go.Figure:
    """Create price comparison bar chart"""
    titles = [p.get("title", f"Product {i+1}")[:30] + "..." for i, p in enumerate(products)]
    prices = [p.get("price_inr", 0) for p in products]
    
    fig = go.Figure(data=[
        go.Bar(
            x=titles,
            y=prices,
            marker_color=['#1f77b4' if i != prices.index(min([p for p in prices if p > 0])) 
                         else '#28a745' for i in range(len(prices))],
            text=[f"₹{p:,.0f}" for p in prices],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Price Comparison",
        xaxis_title="Products",
        yaxis_title="Price (₹)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_rating_comparison_chart(products: List[Dict]) -> go.Figure:
    """Create rating comparison chart"""
    titles = [p.get("title", f"Product {i+1}")[:30] + "..." for i, p in enumerate(products)]
    ratings = [p.get("rating", 0) for p in products]
    
    fig = go.Figure(data=[
        go.Bar(
            x=titles,
            y=ratings,
            marker_color=['#ffc107' if i != ratings.index(max(ratings)) 
                         else '#28a745' for i in range(len(ratings))],
            text=[f"{r:.1f}/5" for r in ratings],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Rating Comparison",
        xaxis_title="Products",
        yaxis_title="Rating (out of 5)",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 5.5])
    )
    
    return fig

def create_value_score_chart(products: List[Dict]) -> go.Figure:
    """Create value score radar chart"""
    best = get_best_product(products)
    if not best:
        return None
    
    categories = ['Price', 'Rating', 'Overall Value']
    
    fig = go.Figure()
    
    for product in products[:3]:  # Show top 3
        rating_score = (product.get("rating", 0) / 5.0) * 100
        max_price = max(p.get("price_inr", 1) for p in products if p.get("price_inr", 0) > 0)
        price_score = (1 - (product.get("price_inr", max_price) / max_price)) * 100
        overall_score = (rating_score * 0.7 + price_score * 0.3)
        
        fig.add_trace(go.Scatterpolar(
            r=[price_score, rating_score, overall_score],
            theta=categories,
            fill='toself',
            name=product.get("title", "Product")[:20]
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Value Score Comparison",
        height=400
    )
    
    return fig


def render_product_card(product: Dict, index: int, is_best: bool = False):
    """Render a product card"""
    title = product.get("title", f"Product {index + 1}")
    price = format_price(product.get("price", "Not available"))
    rating = product.get("rating")
    description = product.get("description", "No description available")
    link = product.get("link", "#")
    
    with st.container():
        if is_best:
            st.markdown("### 🏆 Recommended Choice")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{title}**")
            st.markdown(f'<div class="price-tag">{price}</div>', unsafe_allow_html=True)
        
        with col2:
            if rating is not None:
                st.markdown(f'<div class="rating-badge">⭐ {rating:.1f}/5</div>', unsafe_allow_html=True)
        
        with st.expander("📝 Product Details", expanded=is_best):
            st.write(description)
            features = extract_features(description)
            st.markdown("**Key Features:**")
            for feat in features:
                st.markdown(f"• {feat}")
            st.markdown(f"[🔗 View Product]({link})")

def render_comparison_table(products: List[Dict]):
    """Render comparison table"""
    if not products:
        return
    
    st.markdown("### 📊 Detailed Comparison")
    
    # Prepare data
    rows = ["Price (₹)", "Rating", "Features"]
    cols = [p.get("title", f"Product {i+1}")[:30] + "..." for i, p in enumerate(products)]
    
    data = []
    for row_name in rows:
        row_data = []
        for p in products:
            if row_name == "Price (₹)":
                row_data.append(format_price(p.get("price", "")))
            elif row_name == "Rating":
                rating = p.get("rating")
                row_data.append(f"{rating:.1f}/5 ⭐" if rating else "N/A")
            elif row_name == "Features":
                features = extract_features(p.get("description", ""))[:2]
                row_data.append("\n".join(f"• {f}" for f in features))
        data.append(row_data)
    
    df = pd.DataFrame(data, columns=cols, index=rows)
    st.table(df)

def render_ai_comparison(comparisons: List[Dict]):
    """Render AI-generated comparisons"""
    if not comparisons:
        return
    
    st.markdown("### 🤖 AI Analysis")
    
    for idx, comp in enumerate(comparisons):
        pair = comp.get("pair", ["", ""])
        comparison_text = comp.get("comparison", "")
        
        with st.expander(f"📋 {pair[0]} vs {pair[1]}", expanded=(idx == 0)):
            st.text(comparison_text)

# ====================================
# Main Application
# ====================================
def main():
    # Header
    st.markdown('<div class="main-header">🛒 AI Shopping Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Compare products intelligently with AI-powered analysis</div>',
        unsafe_allow_html=True
    )
    
    
    # Main content
    if not st.session_state.processed:
        # Input Section
        st.markdown("## 🔍 Enter Product URLs")
        
        with st.form("url_form"):
            urls = []
            cols = st.columns(2)
            
            for i in range(st.session_state.url_count):
                col_idx = i % 2
                with cols[col_idx]:
                    url = st.text_input(
                        f"Product URL {i+1}",
                        key=f"url_{i}",
                        placeholder="https://example.com/product"
                    )
                    if url:
                        urls.append(url)
            
            compare_all = st.checkbox(
                "Generate AI Comparison",
                value=True,
                help="Compare all products using AI analysis"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("🚀 Compare Products", use_container_width=True)
            with col2:
                if st.form_submit_button("🔄 Reset", use_container_width=True):
                    st.session_state.processed = False
                    st.session_state.last_result = None
                    st.rerun()
        
        # Process URLs
        if submit:
            # Validate URLs
            valid_urls = [url for url in urls if validate_url(url)]
            
            if not valid_urls:
                st.error("❌ Please enter at least one valid URL")
                return
            
            invalid_urls = [url for url in urls if url and not validate_url(url)]
            if invalid_urls:
                st.warning(f"⚠️ Skipping invalid URLs: {', '.join(invalid_urls)}")
            
            # Prepare payload
            payload = {
                "urls": valid_urls,
                "compare_all": compare_all,
                "use_selenium": True
            }
            
            try:
                if st.session_state.processing_mode == "async":
                    # Async processing
                    st.info("🔄 Starting asynchronous processing...")
                    
                    with st.spinner("Sending request to API..."):
                        response = requests.post(
                            f"{API_URL}/scrape-urls/async",
                            json=payload,
                            timeout=30
                        )
                        response.raise_for_status()
                        result = response.json()
                    
                    task_id = result.get("task_id")
                    st.session_state.task_id = task_id
                    
                    st.success(f"✅ Task created: {task_id}")
                    st.info("Polling task status...")
                    
                    # Poll for results
                    task_result = poll_task_status(task_id)
                    
                    if task_result:
                        st.session_state.last_result = task_result
                        st.session_state.processed = True
                        st.rerun()
                    else:
                        st.error("Failed to get task results")
                
                else:
                    # Synchronous processing
                    with st.spinner(f"🔄 Processing {len(valid_urls)} URLs... This may take a while."):
                        response = requests.post(
                            f"{API_URL}/scrape-urls",
                            json=payload,
                            timeout=300
                        )
                        response.raise_for_status()
                        data = response.json()
                    
                    if data.get("status") == "success":
                        st.session_state.last_result = data
                        st.session_state.processed = True
                        
                        # Save to history
                        st.session_state.comparison_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "products": len(data.get("products", [])),
                            "urls": valid_urls
                        })
                        
                        st.success(f"✅ Successfully processed {len(data.get('products', []))} products!")
                        st.rerun()
                    else:
                        st.error("❌ API returned an error")
            
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try async mode for large batches.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Network error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
    
    else:
        # Results Section
        result = st.session_state.last_result
        products = result.get("products", [])
        comparisons = result.get("comparisons", [])
        
        # Success message
        st.markdown(
            '<div class="success-box">✅ Analysis Complete! Found ' + 
            f'{len(products)} products</div>',
            unsafe_allow_html=True
        )
        
        # Best Product Recommendation
        best_product = get_best_product(products)
        
        if best_product:
            st.markdown("---")
            render_product_card(best_product, 0, is_best=True)
        
    
        # All Products
        st.markdown("---")
        st.markdown("## 📦 All Products")
        
        for idx, product in enumerate(products):
            if best_product and product.get("link") == best_product.get("link"):
                continue  # Skip best product as it's already shown
            render_product_card(product, idx)
        
        # Comparison Table
        if len(products) >= 2:
            st.markdown("---")
            render_comparison_table(products)
        
        # AI Comparisons
        if comparisons:
            st.markdown("---")
            render_ai_comparison(comparisons)
        
        # Export Options
        st.markdown("---")
        st.markdown("## Reset Options")
        
            # Reset button
        if st.button("🔄 New Comparison"):
            st.session_state.processed = False
            st.session_state.last_result = None
            st.session_state.task_id = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666;">'
        'Built with ❤️ using Streamlit, FastAPI & AI | '
        f'<a href="{API_URL}/docs" target="_blank">API Docs</a>'
        '</div>',
        unsafe_allow_html=True
    )

# ====================================
# Run Application
# ====================================
if __name__ == "__main__":
    main()

