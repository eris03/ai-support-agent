import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🤖 AI Support Agent", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stChatMessage { background: white; border-radius: 10px; padding: 15px; margin: 10px 0; }
    h1 { color: white; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
    </style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 AI Support Assistant")
st.markdown("### Powered by Hybrid Search & NLP")
st.markdown("---")

# FAQ Database
faq_data = [
    {"id": 1, "title": "How do I reset my password?", "content": "1. Go to login page\n2. Click 'Forgot Password'\n3. Enter your email\n4. Click reset link in email\n5. Create new password", "category": "Account"},
    {"id": 2, "title": "What is your return policy?", "content": "30-day returns:\n✓ Original condition\n✓ Original packaging\n✓ Proof of purchase\n✓ Full refund minus shipping", "category": "Returns"},
    {"id": 3, "title": "How long does shipping take?", "content": "Standard: 5-7 days\nExpress: 2-3 days\nOvernight: Next day\nInternational: 10-15 days", "category": "Shipping"},
    {"id": 4, "title": "How do I update my payment method?", "content": "Account → Settings → Payment Methods → Add New → Enter card → Save", "category": "Billing"},
    {"id": 5, "title": "What payment methods do you accept?", "content": "Credit cards, Apple Pay, Google Pay, PayPal, Bank transfers", "category": "Billing"},
    {"id": 6, "title": "How do I track my order?", "content": "My Orders → Click order → View tracking → Click for updates", "category": "Shipping"},
    {"id": 7, "title": "How do I contact support?", "content": "Email: support@company.com\nPhone: 1-800-COMPANY\nLive chat: 10am-6pm EST", "category": "General"}
]

# Load Models (Cached)
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faq_texts = [f"{faq['title']} {faq['content']}" for faq in faq_data]
    faq_embeddings = [embedding_model.encode(text, convert_to_tensor=False) for text in faq_texts]
    tokenized = [text.lower().split() for text in faq_texts]
    bm25 = BM25Okapi(tokenized)
    return embedding_model, faq_embeddings, bm25, faq_texts

embedding_model, faq_embeddings, bm25, faq_texts = load_models()

# Search Function
def search_faqs(query):
    query_emb = embedding_model.encode(query, convert_to_tensor=False)
    semantic = [(i, np.dot(query_emb, e) / (np.linalg.norm(query_emb) * np.linalg.norm(e) + 1e-8)) 
                for i, e in enumerate(faq_embeddings)]
    tokenized_q = query.lower().split()
    keyword = [(i, float(s)) for i, s in enumerate(bm25.get_scores(tokenized_q))]
    
    rrf = {}
    for rank, (idx, _) in enumerate(sorted(semantic, key=lambda x: x[1], reverse=True)[:5], 1):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)
    for rank, (idx, _) in enumerate(sorted(keyword, key=lambda x: x[1], reverse=True)[:5], 1):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)
    
    results = []
    for idx, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:3]:
        faq = faq_data[idx]
        results.append({"id": faq["id"], "title": faq["title"], "content": faq["content"], 
                       "category": faq["category"], "confidence": min(score * 1.5, 1.0)})
    return results

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Features")
    st.markdown("✅ Hybrid Search (Semantic + Keyword)\n✅ Confidence Scoring\n✅ Smart Routing\n✅ 7 FAQ Database")
    st.markdown("---")
    st.markdown("### 📞 Support")
    st.markdown("Email: support@company.com\nPhone: 1-800-COMPANY")

# Chat Interface
st.markdown("### 💬 Chat with AI Support")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    faqs = search_faqs(user_input)
    conf = faqs[0]["confidence"] if faqs else 0
    
    with st.chat_message("assistant"):
        if conf >= 0.80:
            response = f"✅ **{faqs[0]['title']}**\n\n{faqs[0]['content']}\n\n📊 Confidence: {conf:.0%} | 📁 {faqs[0]['category']}"
        elif any(kw in user_input.lower() for kw in ["frustrated", "angry", "terrible", "hate"]):
            response = "I understand! A specialist is connecting with you now. 🤝"
        else:
            response = f"📌 **{faqs[0]['title']}**\n\n{faqs[0]['content']}\n\n📊 Confidence: {conf:.0%} | 📁 {faqs[0]['category']}"
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>🚀 Powered by Hybrid Search & NLP | © 2025</div>", unsafe_allow_html=True)
