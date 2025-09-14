import streamlit as st
import json
import numpy as np
import os
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Quote Recommender",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #CAE7D3;
    }
    .stApp > header {
        background-color: #CAE7D3;
    }
    .block-container {
        padding: 2rem;
    }
    .quote-container {
        background-color: #F8FAFC;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .quote-text {
        font-style: italic;
        font-size: 1.1rem;
        color: #333333;
        margin-bottom: 0.75rem;
    }
    .quote-author {
        font-weight: bold;
        color: #555555;
        text-align: right;
    }
    .score {
        font-size: 0.85rem;
        color: #28a745;
        text-align: right;
        margin-top: 0.25rem;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        font-weight: 700;
        font-size: 2.25rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid #adb5bd;
        padding: 0.75rem;
        background-color: white;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading AI model..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_quotes_data():
    try:
        with open("quotes_with_embeddings.json", "r") as f:
            data = json.load(f)
        embeddings = np.array([entry["embedding"] for entry in data])
        return data, embeddings
    except FileNotFoundError:
        st.error("quotes_with_embeddings.json file not found.")
        return None, None

def retrieve_quotes(query, embeddings, data, model, top_k=5):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings)
    indices = np.argsort(similarities[0])[::-1][:top_k]
    results = []
    for idx in indices:
        results.append({
            "quote": data[idx]["quote"],
            "author": data[idx]["author"],
            "tags": data[idx]["tags"],
            "score": similarities[0][idx]
        })
    return results

def get_ai_recommendations(user_query, retrieved_quotes, openai_api_key):
    if not OPENAI_AVAILABLE or not openai_api_key:
        return None
    try:
        client = OpenAI(api_key=openai_api_key)
        context = "Here are some quotes related to your request:\n\n"
        for i, r in enumerate(retrieved_quotes, 1):
            context += f"{i}. \"{r['quote']}\" — {r['author']}\n"
        prompt = f"""
        The user asked: "{user_query}"

        {context}

        Recommend the best matching quotes with brief explanations and suggest top captions.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error with OpenAI API: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="title">Quote Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Find the perfect quotes for any occasion using AI-powered search</p>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        return

    data, embeddings = load_quotes_data()
    if data is None or embeddings is None:
        return

    with st.form("quote_search_form"):
        user_query = st.text_input("What kind of quote are you looking for?", placeholder="e.g., motivation, love, life advice, success...")
        col1, col2 = st.columns([2, 1])
        with col1:
            num_quotes = st.selectbox("Number of quotes to find", [3, 5, 10, 15], index=1)
        with col2:
            use_ai = st.checkbox("AI Analysis", value=True)
        env_api_key = os.getenv('OPENAI_API_KEY')
        if use_ai and env_api_key and OPENAI_AVAILABLE:
            openai_api_key = env_api_key
        else:
            openai_api_key = None
        submitted = st.form_submit_button("🔍 Find Quotes")

    if submitted and user_query:
        with st.spinner("Searching for quotes..."):
            results = retrieve_quotes(user_query, embeddings, data, model, top_k=num_quotes)
            if results:
                st.success(f"✅ Found {len(results)} quotes!")
                if use_ai and openai_api_key:
                    with st.spinner("Getting AI recommendations..."):
                        ai_response = get_ai_recommendations(user_query, results[:3], openai_api_key)
                        if ai_response:
                            st.markdown("### 🤖 AI Recommendations")
                            st.markdown(ai_response)
                            st.markdown("---")
                st.markdown("### 📚 Search Results")
                for i, result in enumerate(results, 1):
                    st.markdown(f"""
                    <div class="quote-container">
                        <div class="quote-text">"{result['quote']}"</div>
                        <div class="quote-author">— {result['author']}</div>
                        <div class="score">Similarity Score: {result['score']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if result.get('tags'):
                        tags = result['tags'] if isinstance(result['tags'], list) else [result['tags']]
                        tag_display = " • ".join([f"#{tag}" for tag in tags if tag])
                        if tag_display:
                            st.markdown(f"**Tags:** {tag_display}")
                    st.markdown("---")
                st.markdown("### 💾 Export Options")
                col1, col2 = st.columns(2)
                with col1:
                    export_text = f"Search Query: {user_query}\n\n"
                    for i, result in enumerate(results, 1):
                        export_text += f"{i}. \"{result['quote']}\" — {result['author']}\n   Score: {result['score']:.3f}\n\n"
                    st.download_button("📄 Download as Text", export_text, file_name=f"quotes_{user_query.replace(' ', '_')}.txt")
                with col2:
                    export_data = {"query": user_query, "results": results}
                    st.download_button("📊 Download as JSON", json.dumps(export_data, indent=2), file_name=f"quotes_{user_query.replace(' ', '_')}.json")
            else:
                st.warning("No quotes found. Try a different search term.")
    elif submitted and not user_query:
        st.error("❌ Please enter a search query.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; font-size: 0.9rem; padding: 1rem;'>
        Built with ❤️ using Streamlit, SentenceTransformers, and OpenAI<br>
        <em>Powered by the English Quotes Dataset</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
