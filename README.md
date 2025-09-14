# Quote Recommendations using a simple RAG System

This is a beginner-friendly project that demonstrates the core concepts of a Retrieval-Augmented Generation (RAG) system in a creative area - to recommend quotes! 
It allows users to search for relevant quotes based on their query and receive a generated explanation using OpenAI's GPT models.

The goal of this project is to implement my understanding of how retrieval and generation work together in a RAG system, without needing complex infrastructure like vector databases.

---

## 🚀 What I've Implemented:

✔ Load a dataset of quotes, authors, and tags  
✔ Embed quotes using a local sentence-transformer model  
✔ Store and reuse embeddings efficiently in a JSON file  
✔ Retrieve the most similar quotes based on cosine similarity  
✔ Use OpenAI’s API model (openai-4o-mini) to generate explanations or insights based on retrieved quotes  
✔ Modular design ready for scaling later with tools like FAISS or Chroma DB

---

## 📂 Dataset

The dataset used here contains:
- `quote`: The text of the quote
- `author`: The person who said it
- `tags`: Categories or themes related to the quote

Link: https://huggingface.co/datasets/Abirate/english_quotes

---

## ⚙ How It Works

1. **Embedding**  
   Each quote is converted into a vector using `sentence-transformers` to capture its meaning.

2. **Retrieval**  
   Given a user query, the system finds the most similar quotes by comparing embeddings using cosine similarity.

3. **Generation**  
   Retrieved quotes are aggregated and passed to OpenAI’s GPT model to generate a context-aware explanation or response.

---

## 🛠 Tech Stack

- **Python**  
- **Streamlit** – Web interface  
- **Sentence Transformers** – Semantic embeddings  
- **scikit-learn** – Similarity calculations  
- **OpenAI GPT (optional)** – Enhanced recommendations and explanations  
- **JSON** – Data storage and export

## Steps: 
1. Clone the repository in Colab 
2. Install required packages as mentioned in requirements.txt
3. Add your OpenAI API KEY in Colab secrets
4. Run the repo and you will get quote recommendations for a user query

   
1. Alternatively, clone the repo on your system, install the required packages.
2. In command prompt, set your API key as environment variable:
   export OPENAI_API_KEY="your_openai_api_key"
3. Run the app.py file which contains your app interface:
   streamlit run app.py

## Streamlit UI
<img width="1920" height="755" alt="image" src="https://github.com/user-attachments/assets/83449445-8701-462c-840d-ecbf5bc2dadb" />

<img width="1891" height="827" alt="image" src="https://github.com/user-attachments/assets/139ad61e-fbb6-4297-b01d-56ed83870b8c" />


