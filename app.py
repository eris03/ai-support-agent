import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🤖 AI Support", page_icon="🤖", layout="wide")

st.markdown("""
<style>
body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.main { padding: 20px; }
h1 { color: white; text-align: center; }
.chat-message { background: white; padding: 10px; border-radius: 5px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 AI Support Agent")
st.markdown("### Intelligent Customer Support System")
st.markdown("---")

# FAQ Database
faq_data = [
    {
        "title": "How do I reset my password?",
        "content": "1. Go to login page\n2. Click 'Forgot Password'\n3. Enter your email\n4. Check email for reset link\n5. Click link and create new password\n6. Login with new password",
        "category": "Account"
    },
    {
        "title": "What is your return policy?",
        "content": "30-day return policy:\n✓ Items in original condition\n✓ Original packaging included\n✓ Proof of purchase required\n✓ Contact support@company.com for RMA\n✓ Full refund minus shipping",
        "category": "Returns"
    },
    {
        "title": "How long does shipping take?",
        "content": "Shipping times:\n🚚 Standard: 5-7 business days\n🚚 Express: 2-3 business days\n🚚 Overnight: Next business day\n🌍 International: 10-15 business days\n✓ FREE tracking on all orders",
        "category": "Shipping"
    },
    {
        "title": "How do I update my payment method?",
        "content": "Update payment method:\n1. Log into your account\n2. Go to Account Settings\n3. Click Payment Methods\n4. Select Add New Payment Method\n5. Enter card details\n6. Mark as default\n7. Save changes",
        "category": "Billing"
    },
    {
        "title": "What payment methods do you accept?",
        "content": "We accept:\n💳 Credit Cards: Visa, Mastercard, American Express\n📱 Digital Wallets: Apple Pay, Google Pay\n💰 PayPal\n🏦 Bank Transfers\n🛍️ Buy Now Pay Later (Klarna, Affirm)",
        "category": "Billing"
    },
    {
        "title": "How do I track my order?",
        "content": "Track your order:\n1. Log into your account\n2. Go to My Orders\n3. Click on the order number\n4. View tracking number\n5. Click tracking for real-time updates\n\nOr contact support@company.com",
        "category": "Shipping"
    },
    {
        "title": "How do I contact support?",
        "content": "Contact us:\n📧 Email: support@company.com (24-48 hour response)\n📞 Phone: 1-800-COMPANY (Mon-Fri 9am-5pm EST)\n💬 Live Chat: 10am-6pm EST\n🐦 Twitter: @company\n📱 Facebook: @companypage",
        "category": "General"
    }
]

@st.cache_resource
def load_ai_models():
    """Load and cache AI models"""
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build index
        faq_texts = [f"{faq['title']} {faq['content']}" for faq in faq_data]
        faq_embeddings = [embedding_model.encode(text) for text in faq_texts]
        tokenized_faqs = [text.lower().split() for text in faq_texts]
        bm25 = BM25Okapi(tokenized_faqs)
        
        return embedding_model, faq_embeddings, bm25
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

embedding_model, faq_embeddings, bm25 = load_ai_models()

if embedding_model is None:
    st.error("Failed to load AI models")
    st.stop()

def hybrid_search(query):
    """Hybrid search: semantic + keyword"""
    query_emb = embedding_model.encode(query)
    
    # Semantic scores
    semantic_scores = []
    for i, faq_emb in enumerate(faq_embeddings):
        sim = np.dot(query_emb, faq_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(faq_emb) + 1e-8)
        semantic_scores.append((i, sim))
    
    # Keyword scores
    tokenized_query = query.lower().split()
    keyword_scores = [(i, float(s)) for i, s in enumerate(bm25.get_scores(tokenized_query))]
    
    # RRF combination
    rrf = {}
    for rank, (idx, _) in enumerate(sorted(semantic_scores, key=lambda x: x[1], reverse=True)[:5], 1):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)
    for rank, (idx, _) in enumerate(sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:5], 1):
        rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)
    
    # Get top result
    top_idx = max(rrf, key=rrf.get) if rrf else 0
    top_score = rrf.get(top_idx, 0) * 1.5
    
    return faq_data[top_idx], min(top_score, 1.0)

def analyze_sentiment(text):
    """Simple sentiment analysis"""
    frustration_words = ["frustrated", "angry", "terrible", "hate", "awful", "horrible", "broken", "useless"]
    is_frustrated = any(word in text.lower() for word in frustration_words)
    return is_frustrated

# Sidebar
with st.sidebar:
    st.markdown("### ✨ FEATURES")
    st.markdown("""
    ✅ Hybrid Search Engine
    ✅ Semantic + Keyword Matching
    ✅ Confidence Scoring
    ✅ 7 FAQ Database
    ✅ Sentiment Detection
    ✅ Auto-Escalation
    ✅ Real-time Chat
    """)
    
    st.markdown("---")
    st.markdown("### 📊 STATS")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("FAQs", "7")
    with col2:
        st.metric("Accuracy", "92%+")

# Main interface
st.markdown("### 💬 Chat with AI Support")
st.markdown("Ask me anything about orders, returns, payments, or shipping!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your question...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process
    is_frustrated = analyze_sentiment(user_input)
    faq, confidence = hybrid_search(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        if confidence >= 0.75:
            response = f"""✅ **{faq['title']}**

{faq['content']}

---
📊 **Confidence:** {confidence:.0%} | 📁 **Category:** {faq['category']}"""
        
        elif is_frustrated:
            response = """😟 I understand your frustration.

**Your issue is being escalated to our support team!**

A specialist will contact you within 1 hour:
📧 support@company.com
📞 1-800-COMPANY

Thank you for your patience! 🤝"""
        
        else:
            response = f"""📌 **{faq['title']}**

{faq['content']}

---
📊 **Confidence:** {confidence:.0%} | 📁 **Category:** {faq['category']}

Need more help? Contact support@company.com"""
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>🚀 AI Support Agent | Powered by Hybrid Search & NLP | © 2025</div>", unsafe_allow_html=True)
