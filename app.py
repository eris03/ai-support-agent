import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🤖 AI Support", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
h1 { color: white; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 AI Support Agent")
st.markdown("### Intelligent Customer Support with Hybrid Search")
st.markdown("---")

# FAQ Data
faqs = [
    {"title": "How do I reset my password?", "content": "Go to login → Forgot Password → Enter email → Check email → Click link → Set new password", "cat": "Account"},
    {"title": "What is your return policy?", "content": "30-day returns: Original condition, original packaging, proof of purchase. Contact support@company.com for RMA.", "cat": "Returns"},
    {"title": "How long does shipping take?", "content": "Standard: 5-7 days | Express: 2-3 days | Overnight: Next day | International: 10-15 days. FREE tracking!", "cat": "Shipping"},
    {"title": "How do I update my payment method?", "content": "Account Settings → Payment Methods → Add New → Enter card details → Mark default → Save", "cat": "Billing"},
    {"title": "What payment methods do you accept?", "content": "Credit cards (Visa, MC, Amex), Apple Pay, Google Pay, PayPal, Bank transfers, Buy now pay later", "cat": "Billing"},
    {"title": "How do I track my order?", "content": "My Orders → Click order → View tracking number → Click for real-time updates", "cat": "Shipping"},
    {"title": "How do I contact support?", "content": "Email: support@company.com | Phone: 1-800-COMPANY | Live chat: 10am-6pm EST", "cat": "General"}
]

@st.cache_resource
def load_ai():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [f"{f['title']} {f['content']}" for f in faqs]
    embeddings = [model.encode(t) for t in texts]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return model, embeddings, bm25

model, embeddings, bm25 = load_ai()

def search(query):
    q_emb = model.encode(query)
    scores = []
    for i, e in enumerate(embeddings):
        sim = np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e) + 1e-8)
        scores.append((i, sim))
    
    kw_scores = [(i, float(s)) for i, s in enumerate(bm25.get_scores(query.lower().split()))]
    
    rrf = {}
    for r, (i, _) in enumerate(sorted(scores, key=lambda x: x[1], reverse=True)[:3], 1):
        rrf[i] = rrf.get(i, 0) + 1/(60+r)
    for r, (i, _) in enumerate(sorted(kw_scores, key=lambda x: x[1], reverse=True)[:3], 1):
        rrf[i] = rrf.get(i, 0) + 1/(60+r)
    
    top_idx = max(rrf, key=rrf.get) if rrf else 0
    return faqs[top_idx], rrf.get(top_idx, 0) * 1.5

# Sidebar
with st.sidebar:
    st.markdown("### ✨ Features")
    st.markdown("✅ Hybrid Search\n✅ Confidence Scoring\n✅ Sentiment Detection\n✅ 7 FAQ Database")

st.markdown("### 💬 Chat with AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    faq, conf = search(user_input)
    frustration = any(w in user_input.lower() for w in ["frustrated", "angry", "terrible", "hate"])
    
    with st.chat_message("assistant"):
        if conf >= 0.75:
            response = f"✅ **{faq['title']}**\n\n{faq['content']}\n\n📊 Confidence: {conf:.0%} | 📁 {faq['cat']}"
        elif frustration:
            response = "I understand your frustration! A specialist is connecting with you now. 🤝"
        else:
            response = f"📌 **{faq['title']}**\n\n{faq['content']}\n\n📊 Confidence: {conf:.0%} | 📁 {faq['cat']}"
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>🚀 AI Support Agent © 2025</div>", unsafe_allow_html=True)
