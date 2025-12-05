# app.py
import os
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Import functions from main (functional style)
from main import create_config  # only if needed; not strictly necessary

load_dotenv()
# API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = "https://ai-product-api.onrender.com"




st.set_page_config(page_title="🛒 AI Shopping Helper — Fresh UI", layout="wide")
st.title("🛒 AI Shopping Helper")
st.write("Enter product URLs compare the product and AI recommended the best product for you .")

# Session state
if "processed" not in st.session_state:
    st.session_state.processed = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "saved_file" not in st.session_state:
    st.session_state.saved_file = ""

def extract_features(text, max_items=3):
    if not text:
        return []
    pieces = [s.strip() for s in text.split('.') if s.strip()]
    features = []
    for p in pieces:
        if len(features) >= max_items:
            break
        if len(p) > 8:
            features.append(p if len(p) < 120 else p[:117] + '...')
    return features or ["Not available"]

# UI: Inputs (hidden after processing)
with st.form("url_form"):
    if not st.session_state.processed:
        cols = st.columns(2)
        with cols[0]:
            url1 = st.text_input("URL 1", key="url1")
        with cols[1]:
            url2 = st.text_input("URL 2", key="url2")
        submit = st.form_submit_button("🚀 Compare URLs ")
    else:
        st.info("Processing completed — inputs are hidden. Use Reset to scan new URLs.")
        submit = False

# Handle submit
if submit and not st.session_state.processed:
    urls = []
    for key in ["url1", "url2"]:
        v = st.session_state.get(key)
        if v and isinstance(v, str) and v.strip():
            urls.append(v.strip())
    if not urls:
        st.warning("Please enter at least one URL to process.")
    else:
        payload = {"urls": urls, "compare_all": True}
        try:
            with st.spinner("Sending URLs to API and processing — this can take a while..."):
                resp = requests.post(f"{API_URL}/compare-two", json=payload, timeout=180)
                resp.raise_for_status()
                data = resp.json()
            if data.get("status") == "success":
                products = data.get("products", [])
                comparisons = data.get("comparisons", [])
                st.session_state.last_result = {"products": products, "comparisons": comparisons}
                out_filename = "streamlit_products.json"
                with open(out_filename, "w", encoding="utf-8") as f:
                    json.dump(products, f, indent=2, ensure_ascii=False)
                st.session_state.saved_file = out_filename
                st.session_state.processed = True
            else:
                st.error("API returned an error or unexpected response.")
        except Exception as e:
            st.error(f"Error calling API: {e}")

# Display results
if st.session_state.last_result:
    result = st.session_state.last_result
    products = result.get("products", [])
    comparisons = result.get("comparisons", [])

    # Product summaries
    st.subheader("📝 Product Summaries")
    for product in products:
        title = product.get("title", "Unnamed Product")
        desc = product.get("description", "")
        link = product.get("link", "")
        price = product.get("price", "Not Available")
        with st.expander(title):
            st.markdown(f"**Price:** {price}")
            st.markdown(f"🔗 [Product Link]({link})")
            st.write("---")
            st.write(desc)

    # Build comparison table with product titles as columns and features as rows
    st.subheader("📊 Product Comparison Table ")
    if products:
        cols = [p.get("title", f"Product {i+1}") for i, p in enumerate(products)]
        rows = ["Price (INR)", "Rating", "Features", "Link"]
        # Build matrix data
        matrix = []
        for r in rows:
            row_vals = []
            for p in products:
                if r == "Price (INR)":
                    row_vals.append(p.get("price", "Not available"))
                elif r == "Rating":
                    row_vals.append(str(p.get("rating", "Not available")))
                elif r == "Features":
                    feats = extract_features(p.get("description", ""), max_items=3)
                    row_vals.append("\n".join(f"- {f}" for f in feats))
                elif r == "Link":
                    row_vals.append(p.get("link", ""))
                else:
                    row_vals.append("")
            matrix.append(row_vals)
        comp_df = pd.DataFrame(matrix, index=rows, columns=cols)
        st.table(comp_df)


    else:
        st.info("No products available to build comparison table.")

    # Show LLM human-readable comparisons (from API)
    if comparisons:
        st.subheader("🔎 AI Comparisons")
        for c in comparisons:
            pair = c.get("pair", ["", ""]) if c.get("pair") else ["", ""]
            with st.expander(f"{pair[0]} vs {pair[1]}"):
                st.text(c.get("comparison", ""))

    # Reset button
    if st.button("🔁 Reset / New Scan"):
        st.session_state.processed = False
        st.session_state.last_result = None
        st.session_state.saved_file = ""
    try:
        if os.path.exists("streamlit_products.json"):
            os.remove("streamlit_products.json")
    except Exception:
        pass
    st.rerun()
else:
    st.write("Waiting for input — enter URLs above and press 'Process URLs via API'.")

st.write("---")
st.caption("Built with ❤️ using Streamlit + FastAPI + AI product extraction")







