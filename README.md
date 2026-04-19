# 📚 Strictly-Cited Machine Learning Textbook Assistant (Advanced RAG)

**It's a problem we all face as students:** Every time a professor assigns a project, they explicitly warn us: *"Don't just depend on AI, you need to rely on the textbook."* We all know the risk—standard LLMs hallucinate, invent facts, or reference external web sources instead of the actual course material.

**So, I engineered a solution to this universal problem.** 

I built a **Production-Grade Retrieval-Augmented Generation (RAG) System** that forces a robust AI model to read, analyze, and cite the exact textbook my professor assigned. Instead of generating blind answers, this assistant is anchored to academic truth, proving its work with **exact page numbers and confidence scores** for every single query.

## 📖 How It Solves The AI Hallucination Problem
When a user asks a question, this system doesn't just ask ChatGPT. It:
1. Intercepts the query and rapidly calculates its mathematical vector.
2. Performs a similarity search against a persistent vector database containing thousands of chunked textbook paragraphs.
3. Retrieves only the most mathematically relevant academic excerpts.
4. Forces the LLM (`llama-3.3-70b-versatile`) to formulate an answer *exclusively* from those excerpts, explicitly returning the page numbers doing so.

## 🛠️ Technical Architecture & Stack
This project was built focusing on modularity, scalability, and clean code architecture.

- **UI Framework & Frontend**: [Streamlit](https://streamlit.io/) for a highly-responsive, chat-based web interface.
- **Inference LLM Engine**: Groq API (`llama-3.3-70b-versatile`) for lightning-fast token generation.
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) deployed locally for persistent, high-performance vector storage and metadata preservation.
- **Embeddings Pipeline**: `sentence-transformers` (`all-MiniLM-L6-v2`) directly integrated via ChromaDB.
- **Data Ingestion**: `PyMuPDFLoader` utilized for robust PDF parsing and automatic page-number/metadata extraction.
- **Orchestration**: [LangChain](https://python.langchain.com/) for recursive chunking, abstraction, and logic orchestration.

## 🚀 Key System Capabilities
- **Strict Sourcing Protocol**: The system architecture mandates that the LLM is fenced into using only textbook context.
- **Verifiable Citations**: Every answer includes an expandable UI element displaying the exact page number and text snippet the AI used to formulate its response.
- **Confidence Scoring Engine**: Exposes the computed cosine-similarity score to the user for maximum transparency.
- **Persistent Local Caching**: Computes embeddings once and caches them to disk (`chroma_db/`), allowing for instant subsequent cold starts.

## ⚙️ Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/thejas-06/RAG.git
cd RAG
```

2. **Install dependencies**
Make sure you have Python 3.12+ installed.
```bash
pip install -r requirements.txt
```

3. **Configure Environment Secrets**
Create `.env` file in the root directory and add your Groq API Key:
```
GROQ_API_KEY="your_groq_api_key_here"
```

4. **Add the Target Asset**
Ensure the target textbook PDF (e.g., `Machine-Learning-TomMitchell.pdf`) is placed inside the root `data/` directory.

5. **Spin up the Application**
```bash
streamlit run app.py
```
*Note: On the first boot, the system will automatically parse the PDF, recursively chunk the context, generate vectors, and populate the Chroma persist directory. Future boots are instantaneous.*

---
*Built not just to leverage AI, but to anchor it to verified knowledge.*
