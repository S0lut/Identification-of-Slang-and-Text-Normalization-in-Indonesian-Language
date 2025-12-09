import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from io import BytesIO
from Levenshtein import ratio
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Hybrid Normalizer", 
    page_icon="", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    st.markdown("""
    <style>
    /* --- GLOBAL DARK THEME --- */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* --- HEADER STYLING --- */
    .main-header {
        background: linear-gradient(90deg, #2b005c 0%, #1e1b4b 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #4c1d95;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.2);
    }
    .main-header h1 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(left, #a78bfa, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* --- GLASSMORPHISM CARDS --- */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: #8b5cf6;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
    }
    
    /* --- METRIC BOXES --- */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        background: #1f2937;
        border-radius: 12px;
        padding: 15px;
        border-bottom: 4px solid #8b5cf6; /* Ungu Neon */
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2dd4bf; /* Teal Neon */
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* --- INPUT FIELDS --- */
    .stTextArea textarea, .stTextInput input {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
        border: 1px solid #374151 !important;
        border-radius: 10px;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 1px #8b5cf6 !important;
    }

    /* --- BUTTONS --- */
    .stButton > button {
        background: linear-gradient(45deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #6d28d9, #1d4ed8);
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.5);
    }

    /* --- RESULT HIGHLIGHTS --- */
    .result-original {
        color: #f87171; /* Merah muda */
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
    }
    .result-normalized {
        color: #4ade80; /* Hijau neon */
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        font-weight: bold;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #374151;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()


class HybridSlangNormalizer:
    def __init__(self, dictionary_path='colloquial-indonesian-lexicon.csv', model_name='indolem/indobert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.slang_dict = self.load_external_dictionary(dictionary_path)
        self.slang_dict.update({
            'tp': 'tetapi', 'tapi': 'tetapi', 'g': 'tidak', 'gak': 'tidak', 
            'ga': 'tidak', 'enggak': 'tidak', 'krn': 'karena', 'cuma': 'hanya', 
            'dr': 'dari', 'bgt': 'banget', 'yg': 'yang', 'ngasih': 'memberi', 
            'kesel': 'kesal', 'bener': 'benar', 'dah': 'sudah', 'udh': 'sudah',
            'klo': 'kalau', 'dimna': 'di mana', 'jga': 'juga', 'smua': 'semua',
            'gw': 'saya', 'gue': 'saya', 'lu': 'kamu', 'lo': 'kamu'
        })
        if not self.slang_dict: self.slang_dict = {'tp': 'tetapi'} 
        self.formal_words = self.build_formal_dictionary()
    
    def load_external_dictionary(self, path):
        try:
            if not os.path.exists(path): return {}
            df = pd.read_csv(path)
            df['slang'] = df['slang'].astype(str)
            df['formal'] = df['formal'].astype(str)
            df = df.drop_duplicates(subset=['slang'])
            return dict(zip(df['slang'], df['formal']))
        except: return {}

    def build_formal_dictionary(self):
        return {
            'saya', 'aku', 'kamu', 'dia', 'kami', 'kita', 'mereka', 'anda',
            'ini', 'itu', 'ada', 'adalah', 'akan', 'sudah', 'belum', 'sedang',
            'yang', 'dengan', 'untuk', 'dari', 'ke', 'di', 'pada', 'kepada',
            'tidak', 'bukan', 'jangan', 'bisa', 'dapat', 'mau', 'ingin',
            'benar', 'salah', 'baik', 'buruk', 'senang', 'sedih', 'marah',
            'jalan', 'makan', 'minum', 'tidur', 'pergi', 'pulang', 'lihat',
            'tetapi', 'namun', 'karena', 'jika', 'sebab', 'agar', 'bagaimana'
        }
    
    def clean_text(self, text):
        text = str(text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'(\w+)2', r'\1 \1', text) 
        text = re.sub(r'[^\w\s-]', ' ', text) 
        text = text.lower().strip()
        return text
    
    def find_slang_words(self, text):
        words = text.split()
        found, unknown = [], []
        for word in words:
            if word in self.slang_dict: found.append(word)
            elif len(word) > 2 and word not in self.formal_words:
                if '-' not in word: unknown.append(word)
        return found, unknown
    
    def get_embedding(self, word):
        inputs = self.tokenizer(word, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def get_batch_embeddings(self, words):
        inputs = self.tokenizer(words, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def find_similar_words(self, word, candidates=None, top_k=5, min_similarity=0.65, alpha=0.6):
        if candidates is None: candidates = list(self.formal_words)
        word_emb = self.get_embedding(word)
        semantic_scores = []
        batch_size = 100
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            batch_embs = self.get_batch_embeddings(batch)
            sims = cosine_similarity(word_emb, batch_embs)[0]
            for j, candidate in enumerate(batch):
                semantic_scores.append((candidate, float(sims[j])))
        final_candidates = []
        for candidate, sem_score in semantic_scores:
            if sem_score < 0.4: continue
            morph_score = ratio(word, candidate)
            final_score = (alpha * sem_score) + ((1 - alpha) * morph_score)
            if final_score >= min_similarity:
                final_candidates.append({'word': candidate, 'final_score': final_score})
        final_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return final_candidates[:top_k]
    
    def normalize_text(self, text):
        cleaned = self.clean_text(text)
        processed = cleaned 
        known, unknown = self.find_slang_words(processed)
        words = processed.split()
        normalized_words = []
        detected_slang = set(known)
        for word in words:
            if word in self.slang_dict:
                normalized_words.append(self.slang_dict[word])
            elif word in unknown:
                similar = self.find_similar_words(word, top_k=1, min_similarity=0.70)
                if similar and similar[0]['final_score'] > 0.70:
                    normalized_words.append(similar[0]['word'])
                    detected_slang.add(word)
                else:
                    normalized_words.append(word)
            else:
                normalized_words.append(word)
        return {
            'original': text, 
            'cleaned': cleaned,
            'detected_slang': list(detected_slang),
            'hybrid_normalized': ' '.join(normalized_words)
        }

def calculate_metrics(predicted_slang_list, true_slang_str, predicted_norm, true_norm_str):
    pred_set = set([s.strip() for s in predicted_slang_list if s.strip()])
    if pd.isna(true_slang_str) or str(true_slang_str).strip() == '':
        true_set = set()
    else:
        true_set = set([s.strip() for s in str(true_slang_str).split(',') if s.strip()])
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set.difference(true_set))
    fn = len(true_set.difference(pred_set))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    union_count = len(pred_set.union(true_set))
    accuracy_detect = tp / union_count if union_count > 0 else (1.0 if not pred_set and not true_set else 0.0)
    ref = [str(true_norm_str).lower().split()]
    cand = str(predicted_norm).lower().split()
    chencherry = SmoothingFunction().method1
    bleu = sentence_bleu(ref, cand, smoothing_function=chencherry)
    return precision, recall, f1, accuracy_detect, bleu


# Header Futuristik
st.markdown("""
<div class="main-header">
    <h1>HYBRID SLANG NORMALIZER</h1> 
    <h5>Normalisasi Teks Slang pada bahasa Indonesia secara instan dengan model IndoBERT & kamus eksternal</h5>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
    st.markdown("###  PANDUAN PENGGUNAAN")

    
    st.markdown("""
    **1️⃣ Tab Input Manual**
    * Masukkan kalimat gaul di kolom teks.
    * Klik tombol **Jalankan Normalisasi**.
    * Centang **Mode Evaluasi** jika ingin memasukkan kunci jawaban untuk melihat akurasi.

    **2️⃣ Tab Batch Upload**
    * Siapkan file **Excel (.xlsx)** atau **CSV**.
    * Upload file pada area yang disediakan.
    * Pilih kolom teks dan kolom evaluasi (jika ada).
    * Klik **Mulai Pemrosesan Batch**.
    * Download hasil di bagian bawah.
    """)
    
    st.markdown("---")
    st.info("💡 **Note:** Pastikan file Excel Anda memiliki header (judul kolom) yang jelas.")


# Load Model
@st.cache_resource
def load_normalizer():
    return HybridSlangNormalizer(dictionary_path='colloquial-indonesian-lexicon.csv')

if 'normalizer' not in st.session_state:
    with st.spinner('🔮 Summoning AI Model...'):
        st.session_state['normalizer'] = load_normalizer()

normalizer = st.session_state['normalizer']

# State
if 'processed_data' not in st.session_state: st.session_state['processed_data'] = None
if 'metrics_report' not in st.session_state: st.session_state['metrics_report'] = None
if 'file_id' not in st.session_state: st.session_state['file_id'] = None

# Tabs
tab1, tab2 = st.tabs(["📝 INPUT MANUAL", "📂 BATCH UPLOAD"])


with tab1:
    col_input, col_tips = st.columns([2, 1])
    
    with col_input:
        user_text = st.text_area(
            "💬 Masukkan Teks Gaul:", 
            placeholder="Ketik di sini (Contoh: gw kesel bgt hari ini...)",
            height=120
        )
    
    with col_tips:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#2dd4bf; margin:0;">💡 Info Model</h4>
            <small style="color:#9ca3af">
            Sistem ini menggabungkan pencarian kamus cepat dan pemahaman konteks IndoBERT untuk hasil terbaik.
            </small>
        </div>
        """, unsafe_allow_html=True)

    # Toggle Evaluasi
    use_evaluation = st.toggle("🎯 Mode Evaluasi (Input Kunci Jawaban)", value=False)
    
    true_norm_input, true_slang_input = "", ""
    if use_evaluation:
        c1, c2 = st.columns(2)
        with c1: true_norm_input = st.text_area("✅ Kalimat Baku (Ground Truth):", height=70)
        with c2: true_slang_input = st.text_input("🏷️ List Slang (Pisah koma):")

    if st.button("✨ JALANKAN NORMALISASI", type="primary", use_container_width=True):
        if not user_text:
            st.warning("⚠️ Input teks kosong!")
        else:
            with st.spinner("🔄 Memproses dengan AI..."):
                res = normalizer.normalize_text(user_text)
            
            st.markdown("### 🔍 Hasil Analisis")
            
            # Kartu Hasil Side-by-Side
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class="glass-card">
                    <p style="color:#9ca3af; margin-bottom:5px;">📥 INPUT ASLI</p>
                    <div class="result-original">""" + res["original"] + """</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="glass-card" style="border-color:#10b981;">
                    <p style="color:#9ca3af; margin-bottom:5px;">📤 HASIL BAKU</p>
                    <div class="result-normalized">""" + res["hybrid_normalized"] + """</div>
                </div>
                """, unsafe_allow_html=True)

            # Info Slang
            slang_found = ", ".join(res['detected_slang']) if res['detected_slang'] else "-"
            st.markdown(f"""
            <div style="background:#3730a3; padding:10px 20px; border-radius:8px; margin-top:10px; border:1px solid #6366f1;">
                👾 <b>Slang Terdeteksi:</b> <span style="color:#2dd4bf">{slang_found}</span>
            </div>
            """, unsafe_allow_html=True)

            # Evaluasi Metrics
            if use_evaluation and true_norm_input and true_slang_input:
                p, r, f1, acc, bleu = calculate_metrics(res['detected_slang'], true_slang_input, res['hybrid_normalized'], true_norm_input)
                
                st.markdown("<br>", unsafe_allow_html=True)
                cols = st.columns(5)
                metrics_data = [
                    ("PRECISION", p, "{:.0%}"), ("RECALL", r, "{:.0%}"), 
                    ("F1 SCORE", f1, "{:.0%}"), ("ACCURACY", acc, "{:.0%}"), ("BLEU", bleu, "{:.3f}")
                ]
                for col, (label, val, fmt) in zip(cols, metrics_data):
                    with col:
                        st.markdown(f"""
                        <div class="metric-container">
                            <span class="metric-value">{fmt.format(val)}</span>
                            <span class="metric-label">{label}</span>
                        </div>
                        """, unsafe_allow_html=True)


with tab2:
    st.markdown("### 📂 Batch File Processor")
    
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['xlsx', 'csv'])
    
    if uploaded_file is None: st.session_state['processed_data'] = None
    elif uploaded_file.file_id != st.session_state['file_id']:
        st.session_state['processed_data'] = None
        st.session_state['file_id'] = uploaded_file.file_id

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            
            st.dataframe(df.head(3), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            text_col = c1.selectbox("Pilih Kolom Teks", df.columns)
            
            idx_s = list(df.columns).index('true_slang') if 'true_slang' in df.columns else 0
            idx_n = list(df.columns).index('true_normalized') if 'true_normalized' in df.columns else 0
            
            col_true_slang = c2.selectbox("Kolom True Slang (Opsional)", [None]+list(df.columns), index=idx_s+1 if 'true_slang' in df.columns else 0)
            col_true_norm = c3.selectbox("Kolom True Norm (Opsional)", [None]+list(df.columns), index=idx_n+1 if 'true_normalized' in df.columns else 0)
            
            if st.button("🚀 MULAI PEMROSESAN BATCH", type="primary"):
                bar = st.progress(0)
                stats = st.empty()
                res_norm, res_slang = [], []
                ps, rs, f1s, accs, bleus = [], [], [], [], []
                
                total = len(df)
                for i, row in df.iterrows():
                    out = normalizer.normalize_text(str(row[text_col]))
                    res_norm.append(out['hybrid_normalized'])
                    res_slang.append(", ".join(out['detected_slang']))
                    
                    if col_true_slang and col_true_norm:
                        p, r, f, a, b = calculate_metrics(out['detected_slang'], row[col_true_slang], out['hybrid_normalized'], row[col_true_norm])
                        ps.append(p); rs.append(r); f1s.append(f); accs.append(a); bleus.append(b)
                    else:
                        ps.append(None)
                    
                    if i % 5 == 0:
                        bar.progress((i+1)/total)
                        stats.text(f"Processing... {int((i+1)/total*100)}%")
                
                bar.progress(1.0)
                stats.text("Selesai!")
                
                df['hybrid_normalized'] = res_norm
                df['detected_slang'] = res_slang
                
                avg = {}
                if col_true_slang and col_true_norm:
                    df['precision'], df['recall'], df['f1'], df['acc'], df['bleu'] = ps, rs, f1s, accs, bleus
                    avg = {
                        'Precision': np.mean([x for x in ps if x is not None]),
                        'Recall': np.mean([x for x in rs if x is not None]),
                        'F1 Score': np.mean([x for x in f1s if x is not None]),
                        'Accuracy': np.mean([x for x in accs if x is not None]),
                        'BLEU': np.mean([x for x in bleus if x is not None])
                    }
                
                st.session_state['processed_data'] = df
                st.session_state['metrics_report'] = avg
                st.rerun()

        except Exception as e: st.error(f"Error: {e}")

    # HASIL
    if st.session_state['processed_data'] is not None:
        res_df = st.session_state['processed_data']
        met = st.session_state['metrics_report']
        
        st.markdown("---")
        if met:
            st.markdown("### 📊 Rata-Rata Batch")
            cols = st.columns(5)
            # Menampilkan metric dengan style baru
            for col, (k, v) in zip(cols, met.items()):
                with col:
                    fmt = "{:.3f}" if k == 'BLEU' else "{:.1%}"
                    st.markdown(f"""
                    <div class="metric-container">
                        <span class="metric-value">{fmt.format(v)}</span>
                        <span class="metric-label">{k}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Slang Frequency Chart")
        

        with plt.style.context("dark_background"):
            all_s = " ".join([str(x) for x in res_df['detected_slang'] if x])
            s_list = [x.strip() for x in all_s.replace(',', ' ').split() if x.strip()]
            
            if s_list:
                cnt = Counter(s_list).most_common(15)
                pdf = pd.DataFrame(cnt, columns=['Slang', 'Count'])
                
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_alpha(0.0) # Transparent bg
                ax.patch.set_alpha(0.0)
                
                sns.barplot(data=pdf, x='Count', y='Slang', palette='viridis', ax=ax)
                ax.set_title("Top 15 Slang Words", color='white', fontsize=14)
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                # Hapus border
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                st.pyplot(fig)
            else:
                st.info("Tidak ada slang ditemukan.")
        
        st.markdown("### 📋 Preview Data")
        st.dataframe(res_df.head(), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download CSV", csv, "result.csv", "text/csv", use_container_width=True)
        with c2:
            out = BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as w: res_df.to_excel(w, index=False)

            st.download_button("📊 Download Excel", out.getvalue(), "result.xlsx", use_container_width=True)
