"""
10_streamlit_app.py — Gujarati Healthcare QA Demo
Run: streamlit run 10_streamlit_app.py
"""
import streamlit as st
import sys, os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gujarati Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); min-height: 100vh; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(99,102,241,0.3);
    }
    .hero-header h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0; }
    .hero-header p  { color: rgba(255,255,255,0.80); font-size: 1rem; margin: 0.25rem 0 0; }

    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(99,102,241,0.4);
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .answer-card h3 { color: #a5b4fc; font-size: 0.85rem; text-transform: uppercase;
                       letter-spacing: 0.1em; margin-bottom: 0.75rem; }
    .answer-text { color: #f1f5f9; font-size: 1.05rem; line-height: 1.7; }

    /* Emergency card */
    .emergency-card {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.4)} 50%{box-shadow:0 0 0 12px rgba(239,68,68,0)} }
    .emergency-card h3 { color: #fca5a5; }
    .emergency-card p  { color: #fef2f2; font-size: 1.05rem; }

    /* Context card */
    .context-card {
        background: #0f172a;
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 0.75rem;
    }
    .context-item {
        background: rgba(30,41,59,0.8);
        border-left: 3px solid #8b5cf6;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .kg-badge {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .disease-badge {
        background: rgba(239,68,68,0.15);
        color: #fca5a5;
        border-color: rgba(239,68,68,0.4);
    }
    .treatment-badge {
        background: rgba(34,197,94,0.15);
        color: #86efac;
        border-color: rgba(34,197,94,0.4);
    }

    /* Input styling */
    .stTextArea textarea {
        background: #1e293b !important;
        border: 2px solid rgba(99,102,241,0.4) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
    }
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0f172a !important;
    }
    .stSlider [data-baseweb="slider"] { color: #6366f1; }
    div[data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 10px;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Pipeline (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading Gujarati Healthcare AI (takes 1–2 min)...")
def load_pipeline():
    try:
        from pipeline import answer
        return answer
    except Exception as e:
        return None, str(e)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Retrieved Documents (top-k)", 1, 10, 5)
    max_tokens = st.slider("Max Response Tokens", 100, 500, 350, 50)

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "ડાયાબિટીઝ ના લક્ષણો અને ઉપાય",
        "ઉચ્ચ blood pressure ઘટાડવા",
        "Paracetamol ક્યારે લઈ શકાય?",
        "dengue fever symptoms",
        "ઘૂંટણ ના દુખાવા ની સારવાર",
        "ઊંઘ ન આવવી - Insomnia",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:15]}", use_container_width=True):
            st.session_state.query_input = ex

    st.markdown("---")
    st.markdown("""
    <div style="color: #64748b; font-size: 0.8rem; text-align: center;">
    🏥 Gujarati Healthcare QA<br>
    Qwen 2.5 2B · QLoRA · ChromaDB<br>
    Built with ❤️ for Gujarati speakers
    </div>
    """, unsafe_allow_html=True)

# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🏥 ગુજરાતી આરોગ્ય સહાયક</h1>
    <p>Gujarati Healthcare Assistant · Powered by Qwen 2.5 2B + QLoRA + Hybrid RAG</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "Qwen 2.5 2B")
col2.metric("Fine-tuning", "QLoRA (4-bit)")
col3.metric("Retrieval", "Hybrid RAG")
col4.metric("Language", "Gujarati 🇮🇳")

st.markdown("---")

# ─── Query Input ───────────────────────────────────────────────────────────────
query = st.text_area(
    "💬 તમારો પ્રશ્ન અહीं ટાઈપ કરો (Type your health question in Gujarati or English):",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="ઉદા. 'ડાયાબિટીઝ ના લક્ષણો ક્યા છે?' / 'What are diabetes symptoms?'",
    key="query_main"
)

ask_btn = st.button("🔍 Ask / પૂછો", type="primary")

# ─── Answer Section ────────────────────────────────────────────────────────────
if ask_btn and query.strip():
    pipeline_fn = load_pipeline()

    if pipeline_fn is None:
        st.error("❌ Pipeline failed to load. Make sure Notebooks 5–8 have been run.")
    else:
        with st.spinner("🤔 Thinking in Gujarati..."):
            result = pipeline_fn(query.strip(), top_k=top_k, max_new_tokens=max_tokens)

        # Display answer
        if result.get("is_emergency"):
            st.markdown(f"""
            <div class="emergency-card">
                <h3>⚠️ EMERGENCY — તાત્કાલિક સ્થિતિ</h3>
                <p>{result['answer'].replace(chr(10), '<br>')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="answer-card">
                <h3>🏥 Gujarati Healthcare Answer</h3>
                <div class="answer-text">{result['answer'].replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Retrieved context
        with st.expander("📚 Retrieved Context (Vector DB + KG)", expanded=False):
            if result.get("vector_results"):
                st.markdown("**📄 Relevant Passages:**")
                for i, doc in enumerate(result["vector_results"][:3]):
                    st.markdown(f"""<div class="context-item">[{i+1}] (relevance: {doc.get('score', 0):.2f}) {doc['text'][:200]}...</div>""",
                                unsafe_allow_html=True)

            kg = result.get("kg_results", {})
            if kg.get("matched_entities") or kg.get("possible_diseases") or kg.get("suggested_treatments"):
                st.markdown("**🕸️ Knowledge Graph Insights:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**Identified:**")
                    for e in kg.get("matched_entities", [])[:5]:
                        st.markdown(f'<span class="kg-badge">{e}</span>', unsafe_allow_html=True)
                with col_b:
                    st.markdown("**Conditions:**")
                    for d in kg.get("possible_diseases", [])[:5]:
                        st.markdown(f'<span class="kg-badge disease-badge">{d}</span>', unsafe_allow_html=True)
                with col_c:
                    st.markdown("**Treatments:**")
                    for t in kg.get("suggested_treatments", [])[:5]:
                        st.markdown(f'<span class="kg-badge treatment-badge">{t}</span>', unsafe_allow_html=True)

        # Safety notice
        st.info("ℹ️ **Disclaimer:** This assistant is for educational purposes only. Always consult a qualified doctor for medical advice. | આ સહાયક ફક્ત શૈક્ષણિક ઉપયોગ માટે છે. સ્વાસ્થ્ય સમસ્યા માટે ડૉક્ટરની સલાહ ફરજિયાત છે.")

elif ask_btn and not query.strip():
    st.warning("⚠️ Please type a question first.")
