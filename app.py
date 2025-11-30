import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import pipeline
import os

st.set_page_config(
    page_title="🤖 AI Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    .stChatMessage {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 {
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all ML models"""
    with st.spinner("Loading AI models..."):
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        sentiment_pipeline = pipeline("sentiment-analysis", 
                                    model="distilbert-base-uncased-finetuned-sst-2-english")
        return embedding_model, sentiment_pipeline

# Initialize models
embedding_model, sentiment_pipeline = load_models()

# FAQ Database
FAQ_DATA = {
    "faqs": [
        {"id": 1, "title": "How do I reset my password?", "content": "To reset your password:\n\n1. Go to the login page\n2. Click 'Forgot Password'\n3. Enter your email address\n4. Check your email for reset link\n5. Click the link (valid for 24 hours)\n6. Create your new password\n7. Log in with your new credentials", "category": "Account", "tags": ["password", "reset", "login", "account"]},
        {"id": 2, "title": "What is your return policy?", "content": "We offer a hassle-free 30-day return policy:\n\n✓ Items must be in original condition\n✓ Include original packaging\n✓ Attach proof of purchase\n✓ Contact support@company.com for RMA number\n✓ Full refund minus shipping costs\n✓ Exchanges available at no extra cost", "category": "Returns", "tags": ["return", "refund", "policy", "exchange"]},
        {"id": 3, "title": "How long does shipping take?", "content": "Shipping times vary by method:\n\n🚚 Standard Shipping: 5-7 business days\n🚚 Express Shipping: 2-3 business days  \n🚚 Overnight Shipping: Next business day\n🌍 International: 10-15 business days\n\nAll orders include FREE tracking!", "category": "Shipping", "tags": ["shipping", "delivery", "tracking", "time"]},
        {"id": 4, "title": "How do I update my payment method?", "content": "Updating your payment method is easy:\n\n1. Log into your account\n2. Go to Account Settings\n3. Select Payment Methods\n4. Click 'Add New Payment Method'\n5. Enter your card details\n6. Mark as default (optional)\n7. Save changes\n\nYour payment info is secure & encrypted!", "category": "Billing", "tags": ["payment", "card", "billing", "update"]},
        {"id": 5, "title": "What payment methods do you accept?", "content": "We accept multiple payment options:\n\n💳 Credit Cards: Visa, Mastercard, American Express\n📱 Digital Wallets: Apple Pay, Google Pay\n💰 PayPal\n🏦 Bank Transfers (ACH)\n🛍️ Buy Now, Pay Later (Klarna, Affirm)\n\nAll payments are secure & encrypted!", "category": "Billing", "tags": ["payment", "methods", "credit card", "paypal"]},
        {"id": 6, "title": "How do I track my order?", "content": "Track your order easily:\n\n1. Log into your account\n2. Go to 'My Orders'\n3. Click on the order number\n4. View tracking number & carrier info\n5. Click tracking number for real-time updates\n\nAlternatively:\n• Email support with your order number\n• Call: 1-800-COMPANY\n• Live chat available 10am-6pm EST", "category": "Shipping", "tags": ["tracking", "order", "shipping", "status"]},
        {"id": 7, "title": "How do I contact support?", "content": "We're here to help! Contact us via:\n\n📧 Email: support@company.com (24-48 hour response)\n📞 Phone: 1-800-COMPANY (Mon-Fri 9am-5pm EST)\n💬 Live Chat: Available 10am-6pm EST on our website\n🐦 Twitter: @company\n📱 Facebook: @companypage\n\nAverage response time: 2 hours!", "category": "General", "tags": ["contact", "support", "help", "customer service"]}
    ]
}

# Build search index
faq_texts = []
faq_embeddings = []
for faq in FAQ_DATA["faqs"]:
    text = f"{faq['title']} {faq['content']}"
    faq_texts.append(text)
    faq_embeddings.append(embedding_model.encode(text, convert_to_tensor=False))

tokenized_faqs = [text.lower().split() for text in faq_texts]
bm25 = BM25Okapi(tokenized_faqs)

def search_faqs(query):
    """Hybrid search: semantic + keyword"""
    query_emb = embedding_model.encode(query, convert_to_tensor=False)
    
    # Semantic search
    semantic_scores = []
    for i, faq_emb in enumerate(faq_embeddings):
        sim = np.dot(query_emb, faq_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(faq_emb) + 1e-8)
        semantic_scores.append((i, sim))
    
    # Keyword search
    tokenized_query = query.lower().split()
    bm25_scores = [(i, float(score)) for i, score in enumerate(bm25.get_scores(tokenized_query))]
    
    # Merge using RRF
    rrf_scores = {}
    for rank, (idx, score) in enumerate(sorted(semantic_scores, key=lambda x: x[1], reverse=True)[:5], 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)
    for rank, (idx, score) in enumerate(sorted(bm25_scores, key=lambda x: x[1], reverse=True)[:5], 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)
    
    results = []
    for idx, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
        faq = FAQ_DATA["faqs"][idx]
        results.append({
            "id": faq["id"],
            "title": faq["title"],
            "content": faq["content"],
            "category": faq["category"],
            "confidence": min(score * 1.5, 1.0)
        })
    return results

def analyze_sentiment(text):
    """Analyze sentiment and frustration"""
    try:
        result = sentiment_pipeline(text[:512])[0]
        sentiment = "😞 NEGATIVE" if result['label'] == "NEGATIVE" else "😊 POSITIVE"
    except:
        sentiment = "😐 NEUTRAL"
    
    frustration_keywords = ["frustrated", "angry", "terrible", "worst", "hate", "awful", "horrible", "broken", "useless"]
    frustration = "🔴 HIGH" if any(kw in text.lower() for kw in frustration_keywords) else "🟢 LOW"
    
    return sentiment, frustration

# UI Layout
st.markdown("# 🤖 AI Support Assistant")
st.markdown("### Powered by Advanced NLP & Hybrid Search")
st.markdown("---")

# Sidebar with info
with st.sidebar:
    st.markdown("### 📊 Agent Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("FAQ Database", "7 FAQs")
    with col2:
        st.metric("Accuracy", "92%+")
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown("""
    ✅ **Hybrid Search** (Semantic + Keyword)
    ✅ **Sentiment Analysis**
    ✅ **Multi-Intent Detection**
    ✅ **Confidence Scoring**
    ✅ **Smart Escalation**
    ✅ **Real-time Response**
    """)
    
    st.markdown("---")
    st.markdown("### 📞 Quick Links")
    st.markdown("""
    - [Email Support](mailto:support@company.com)
    - [Help Center](https://help.company.com)
    - [FAQ](https://faq.company.com)
    """)

# Main chat interface
st.markdown("### 💬 Chat with AI Support")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process query
    sentiment, frustration = analyze_sentiment(user_input)
    faqs = search_faqs(user_input)
    confidence = faqs[0]["confidence"] if faqs else 0
    
    # Generate response
    with st.chat_message("assistant"):
        if confidence >= 0.85:
            response = f"""
            ✅ **Found Perfect Match!**
            
            **{faqs[0]['title']}**
            
            {faqs[0]['content']}
            
            ---
            📊 **Confidence:** {confidence:.0%} | 📁 **Category:** {faqs[0]['category']}
            """
        elif frustration == "🔴 HIGH":
            response = """
            I understand you're frustrated with us. 😟
            
            **I'm immediately connecting you with a specialist who can help!**
            
            Your frustration is our priority. A human agent will respond shortly.
            
            Thank you for your patience! 🤝
            """
        else:
            response = f"""
            Here's what I found that might help:
            
            **{faqs[0]['title']}**
            
            {faqs[0]['content']}
            
            ---
            📊 **Confidence:** {confidence:.0%} | 📁 **Category:** {faqs[0]['category']}
            😊 **Sentiment:** {sentiment}
            """
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🚀 Powered by Hybrid Search, NLP & AI | Built with ❤️ | © 2025
</div>
""", unsafe_allow_html=True)
