"""
app/streamlit_app.py — Gujarati Healthcare Assistant (GraphRAG Edition)
Run: streamlit run app/streamlit_app.py
"""
import sys
import os
# Add project root to path so `src` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Gujarati Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
.hero { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(99,102,241,0.3); }
.hero h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0; }
.hero p  { color: rgba(255,255,255,.8); margin: .25rem 0 0; }
.answer-card { background: linear-gradient(135deg,#1e293b,#0f172a);
               border: 1px solid rgba(99,102,241,.4); border-left: 4px solid #6366f1;
               border-radius: 12px; padding: 1.5rem; margin-top: 1rem;
               box-shadow: 0 4px 24px rgba(0,0,0,.3); }
.answer-card h3 { color: #a5b4fc; font-size:.85rem; text-transform:uppercase;
                  letter-spacing:.1em; margin-bottom:.75rem; }
.answer-text { color:#f1f5f9; font-size:1.05rem; line-height:1.7; }
.emergency-card { background:linear-gradient(135deg,#7f1d1d,#991b1b);
                  border:1px solid #ef4444; border-left:4px solid #ef4444;
                  border-radius:12px; padding:1.5rem; margin-top:1rem;
                  animation:pulse 2s infinite; }
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4)}50%{box-shadow:0 0 0 12px rgba(239,68,68,0)}}
.emergency-card h3{color:#fca5a5;} .emergency-card p{color:#fef2f2;font-size:1.05rem;}
.cache-badge { display:inline-block; background:rgba(34,197,94,.15); color:#86efac;
               border:1px solid rgba(34,197,94,.4); border-radius:20px;
               padding:.2rem .75rem; font-size:.8rem; }
.kg-badge { display:inline-block; background:rgba(99,102,241,.2); color:#a5b4fc;
            border:1px solid rgba(99,102,241,.4); border-radius:20px;
            padding:.2rem .6rem; font-size:.78rem; margin:.15rem; }
.context-item { background:rgba(30,41,59,.8); border-left:3px solid #8b5cf6;
                padding:.6rem 1rem; border-radius:8px; margin-bottom:.4rem;
                color:#cbd5e1; font-size:.88rem; line-height:1.5; }
.stTextArea textarea { background:#1e293b !important; border:2px solid rgba(99,102,241,.4) !important;
                        border-radius:10px !important; color:#f1f5f9 !important; font-size:1rem !important; }
.stButton > button { background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
                     color:#fff !important; border:none !important; border-radius:10px !important;
                     font-weight:600 !important; width:100%; transition:transform .2s,box-shadow .2s !important; }
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 6px 20px rgba(99,102,241,.4) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load pipeline (cached singleton) ─────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading Gujarati Healthcare AI…")
def load_pipeline():
    try:
        from src.pipeline.inference import MedicalPipeline
        return MedicalPipeline()
    except Exception as e:
        return None, str(e)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Retrieved Documents (top-k)", 1, 10, 5)
    max_tokens = st.slider("Max Response Tokens", 100, 500, 350, 50)
    show_graph = st.checkbox("Show Knowledge Graph results", value=True)
    show_context = st.checkbox("Show retrieved passages", value=True)

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "ડાયાબિટીઝ ના લક્ષણો અને ઉપાય",
        "ઉચ્ચ blood pressure ઘટાડવા",
        "Paracetamol ક્યારે લઈ શકાય?",
        "dengue fever symptoms",
        "ઘૂંટણ ના દુખાવા ની સારવાર",
        "heart attack symptoms emergency",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:15]}", use_container_width=True):
            st.session_state.query_input = ex

    st.markdown("---")
    st.markdown("""
    <div style="color:#64748b;font-size:.8rem;text-align:center;">
    🏥 Gujarati Healthcare SLM<br>
    Qwen 2.5 3B · QLoRA · GraphRAG<br>
    Neo4j · Redis · ChromaDB
    </div>
    """, unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏥 ગુજરાતી આરોગ્ય સહાયક</h1>
  <p>Gujarati Healthcare Assistant · Qwen 2.5 3B + QLoRA + GraphRAG (Neo4j + Redis + ChromaDB)</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "Qwen 2.5 3B")
col2.metric("Fine-tuning", "QLoRA (4-bit)")
col3.metric("Retrieval", "GraphRAG")
col4.metric("Graph DB", "Neo4j 🕸️")

st.markdown("---")

query = st.text_area(
    "💬 તમારો પ્રશ્ન ટાઈપ કરો (Gujarati or English):",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="ઉદા. 'ડાયાબિટીઝ ના લક્ષણો?' / 'What are diabetes symptoms?'",
    key="query_main",
)
ask_btn = st.button("🔍 Ask / પૂછો", type="primary")

if ask_btn and query.strip():
    pipeline = load_pipeline()
    if pipeline is None:
        st.error("❌ Pipeline failed to load.")
    else:
        with st.spinner("🤔 Thinking…"):
            result = pipeline.answer(query.strip(), top_k=top_k, max_new_tokens=max_tokens)

        # Cache badge
        if result.get("cache_hit"):
            st.markdown('<span class="cache-badge">⚡ Redis Cache Hit</span>', unsafe_allow_html=True)

        # Answer
        if result.get("is_emergency"):
            st.markdown(f"""
            <div class="emergency-card">
              <h3>⚠️ EMERGENCY — તાત્કાલિક સ્થિતિ</h3>
              <p>{result['answer'].replace(chr(10), '<br>')}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="answer-card">
              <h3>🏥 Gujarati Healthcare Answer</h3>
              <div class="answer-text">{result['answer'].replace(chr(10), '<br>')}</div>
            </div>""", unsafe_allow_html=True)

        # Knowledge Graph insights
        if show_graph:
            kg = result.get("kg_results", {})
            if any(kg.get(k) for k in ("possible_diseases", "suggested_treatments", "suggested_drugs")):
                with st.expander("🕸️ Knowledge Graph Insights (Neo4j)", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**Diseases**")
                        for d in kg.get("possible_diseases", [])[:6]:
                            st.markdown(f'<span class="kg-badge">{d}</span>', unsafe_allow_html=True)
                    with c2:
                        st.markdown("**Treatments**")
                        for t in kg.get("suggested_treatments", [])[:6]:
                            st.markdown(f'<span class="kg-badge">{t}</span>', unsafe_allow_html=True)
                    with c3:
                        st.markdown("**Drugs**")
                        for dr in kg.get("suggested_drugs", [])[:6]:
                            st.markdown(f'<span class="kg-badge">{dr}</span>', unsafe_allow_html=True)

        # Vector passages
        if show_context and result.get("vector_results"):
            with st.expander("📚 Retrieved Medical Book Passages (ChromaDB)"):
                for i, doc in enumerate(result["vector_results"][:3]):
                    st.markdown(
                        f'<div class="context-item">[{i+1}] <b>{doc["source"]}</b> '
                        f'(score={doc["score"]:.2f})<br>{doc["text"][:300]}…</div>',
                        unsafe_allow_html=True,
                    )

        st.info("ℹ️ **Disclaimer:** Educational purposes only. Always consult a doctor.")

elif ask_btn and not query.strip():
    st.warning("⚠️ Please type a question first.")
