import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings('ignore')

# Page Config
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
}
.stChatMessage {
    background: white;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
h1 {
    color: white;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.metric-box {
    background: white;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🤖 AI Support Agent")
st.markdown("### Powered by Hybrid Search & NLP")
st.markdown("---")

# FAQ Database
FAQ_DATABASE = [
    {
        "id": 1,
        "title": "How do I reset my password?",
        "content": "To reset your password:\n1. Go to the login page\n2. Click 'Forgot Password'\n3. Enter your email address\n4. Check your email for reset link\n5. Click the link and create a new password\n6. Log in with your new credentials",
        "category": "Account",
        "tags": ["password", "reset", "login", "account"]
    },
    {
        "id": 2,
        "title": "What is your return policy?",
        "content": "We offer a hassle-free 30-day return policy:\n✓ Items must be in original condition\n✓ Include original packaging\n✓ Attach proof of purchase\n✓ Contact support@company.com for RMA\n✓ Full refund minus shipping costs",
        "category": "Returns",
        "tags": ["return", "refund", "policy", "exchange"]
    },
    {
        "id": 3,
        "title": "How long does shipping take?",
        "content": "Shipping times vary by method:\n🚚 Standard: 5-7 business days\n🚚 Express: 2-3 business days\n🚚 Overnight: Next business day\n🌍 International: 10-15 business days\n✓ All orders include FREE tracking",
        "category": "Shipping",
        "tags": ["shipping", "delivery", "tracking", "time"]
    },
    {
        "id": 4,
        "title": "How do I update my payment method?",
        "content": "Updating your payment method is easy:\n1. Log into your account\n2. Go to Account Settings\n3. Select Payment Methods\n4. Click 'Add New Payment Method'\n5. Enter your card details\n6. Mark as default (optional)\n7. Save changes",
        "category": "Billing",
        "tags": ["payment", "card", "billing", "update"]
    },
    {
        "id": 5,
        "title": "What payment methods do you accept?",
        "content": "We accept multiple payment options:\n💳 Credit Cards: Visa, Mastercard, American Express\n📱 Digital Wallets: Apple Pay, Google Pay\n💰 PayPal\n🏦 Bank Transfers (ACH)\n🛍️ Buy Now, Pay Later (Klarna, Affirm)",
        "category": "Billing",
        "tags": ["payment", "methods", "credit card", "paypal"]
    },
    {
        "id": 6,
        "title": "How do I track my order?",
        "content": "Track your order easily:\n1. Log into your account\n2. Go to 'My Orders'\n3. Click on the order number\n4. View tracking number & carrier info\n5. Click tracking number for real-time updates\n\nAlternatively:\n• Email support with order number\n• Call: 1-800-COMPANY\n• Live chat available 10am-6pm EST",
        "category": "Shipping",
        "tags": ["tracking", "order", "shipping", "status"]
    },
    {
        "id": 7,
        "title": "How do I contact support?",
        "content": "We're here to help! Contact us via:\n📧 Email: support@company.com (24-48 hour response)\n📞 Phone: 1-800-COMPANY (Mon-Fri 9am-5pm EST)\n💬 Live Chat: Available 10am-6pm EST on our website\n🐦 Twitter: @company\n📱 Facebook: @companypage",
        "category": "General",
        "tags": ["contact", "support", "help", "customer service"]
    }
]

# Load Models (Cached for speed)
@st.cache_resource
def load_models():
    """Load AI models once and cache them"""
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build search index
        faq_texts = [f"{faq['title']} {faq['content']}" for faq in FAQ_DATABASE]
        faq_embeddings = [embedding_model.encode(text, convert_to_tensor=False) for text in faq_texts]
        tokenized_faqs = [text.lower().split() for text in faq_texts]
        bm25 = BM25Okapi(tokenized_faqs)
        
        return embedding_model, faq_embeddings, bm25, faq_texts
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Load models
embedding_model, faq_embeddings, bm25, faq_texts = load_models()

if embedding_model is None:
    st.error("Failed to load AI models. Please refresh the page.")
    st.stop()

# Hybrid Search Function
def hybrid_search(query):
    """Perform hybrid search: semantic + keyword"""
    try:
        query_emb = embedding_model.encode(query, convert_to_tensor=False)
        
        # Semantic search scores
        semantic_scores = []
        for i, faq_emb in enumerate(faq_embeddings):
            sim = np.dot(query_emb, faq_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(faq_emb) + 1e-8)
            semantic_scores.append((i, sim))
        
        # Keyword search scores (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = [(i, float(score)) for i, score in enumerate(bm25.get_scores(tokenized_query))]
        
        # Reciprocal Rank Fusion (RRF) - Combine both scores
        rrf_scores = {}
        for rank, (idx, score) in enumerate(sorted(semantic_scores, key=lambda x: x[1], reverse=True)[:5], 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)
        for rank, (idx, score) in enumerate(sorted(bm25_scores, key=lambda x: x[1], reverse=True)[:5], 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)
        
        # Get top results
        results = []
        for idx, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            faq = FAQ_DATABASE[idx]
            confidence = min(score * 1.5, 1.0)
            results.append({
                "id": faq["id"],
                "title": faq["title"],
                "content": faq["content"],
                "category": faq["category"],
                "confidence": confidence
            })
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# Sentiment Analysis
def analyze_sentiment(text):
    """Simple sentiment analysis"""
    frustration_keywords = ["frustrated", "angry", "terrible", "worst", "hate", "awful", "horrible", "broken", "useless", "sucks"]
    frustration = "HIGH" if any(kw in text.lower() for kw in frustration_keywords) else "LOW"
    return frustration

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Agent Features")
    st.markdown("""
    ✅ **Hybrid Search**
    Combines semantic & keyword search
    
    ✅ **Confidence Scoring**
    Shows match reliability
    
    ✅ **Smart Routing**
    Routes to FAQ or escalation
    
    ✅ **7 FAQ Database**
    Pre-loaded common questions
    """)
    
    st.markdown("---")
    st.markdown("### 📞 Support Channels")
    st.markdown("""
    📧 Email: support@company.com
    📞 Phone: 1-800-COMPANY
    💬 Live Chat: 10am-6pm EST
    """)

# Main Chat Interface
st.markdown("### 💬 Chat with AI Support")
st.markdown("Ask me anything about orders, returns, payments, or shipping!")

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
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process query
    frustration = analyze_sentiment(user_input)
    search_results = hybrid_search(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        if search_results and search_results[0]["confidence"] >= 0.75:
            # High confidence FAQ match
            faq = search_results[0]
            response = f"""✅ **{faq['title']}**

{faq['content']}

---
📊 **Confidence:** {faq['confidence']:.0%} | 📁 **Category:** {faq['category']}"""
        
        elif frustration == "HIGH":
            # Escalate frustrated customer
            response = """😟 I understand your frustration.

**Your issue is being escalated to our support team immediately!**

A specialist will contact you within 1 hour:
📧 support@company.com
📞 1-800-COMPANY

We appreciate your patience! 🤝"""
        
        elif search_results:
            # Medium confidence match
            faq = search_results[0]
            response = f"""📌 **{faq['title']}**

{faq['content']}

---
📊 **Confidence:** {faq['confidence']:.0%} | 📁 **Category:** {faq['category']}

Didn't find what you need? Contact support@company.com"""
        
        else:
            # No match found
            response = """I couldn't find the answer to your question in our FAQ.

Please contact our support team:
📧 **Email:** support@company.com
📞 **Phone:** 1-800-COMPANY (Mon-Fri 9am-5pm EST)
💬 **Live Chat:** Available 10am-6pm EST

We're here to help! 🤝"""
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
🚀 Powered by Hybrid Search & NLP | AI Support Agent © 2025
</div>
""", unsafe_allow_html=True)
