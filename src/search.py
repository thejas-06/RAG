import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL_NAME, DEFAULT_TOP_K, DEFAULT_MIN_SCORE
from src.vectorstore import ChromaDBStore

load_dotenv()

class AdvancedRAGPipeline:
    def __init__(self):
        self.vector_store = ChromaDBStore()
        
        # Build Vector Store on Initialization if Empty
        if self.vector_store.collection.count() == 0:
            print("[INFO] Vector database is empty. Ingesting textbook data...")
            from src.data_loader import load_documents
            docs = load_documents()
            self.vector_store.build_from_documents(docs)
            
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or groq_api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is not set in the .env file. Please check .env.example")
            
        self.llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=LLM_MODEL_NAME,
            temperature=0.1,
            max_tokens=1024
        )
        print(f"[INFO] Groq LLM initialized: {LLM_MODEL_NAME}")
        self.history = []

    def query(self, question: str, top_k: int = DEFAULT_TOP_K, min_score: float = DEFAULT_MIN_SCORE) -> Dict[str, Any]:
        """
        Retrieves relevant textbook context and generates an answer alongside exact page citations.
        """
        results = self.vector_store.query(question, top_k=top_k, score_threshold=min_score)
        
        if not results:
            answer = "I could not find relevant information in the textbook to answer this question."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source', 'unknown').split('/')[-1].split('\\')[-1],
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:150] + '...'
            } for doc in results]
            
            prompt = f"""You are an advanced Machine Learning Textbook Assistant. Use the following textbook excerpts to answer the student's question. 

Crucial Instructions:
1. Base your answer EXCLUSIVELY on the Context provided below.
2. Pay attention to the requested length. If the student asks for a "15 mark" summary or "detailed" explanation, you MUST provide a comprehensive, highly detailed, multi-paragraph response with bullet points covering as much of the Context as possible.
3. If it's a standard question, answer clearly and accurately.

Context:
{context}

Question: {question}

Answer:"""
            
            response = self.llm.invoke([prompt])
            answer = response.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources
        })

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'history': self.history
        }

if __name__ == "__main__":
    adv_rag = AdvancedRAGPipeline()
    result = adv_rag.query("What is Machine Learning?", top_k=3, min_score=0.1)
    print("\n--- Final Answer ---")
    print(result['answer'])
    print("\n--- Citations ---")
    for s in result['sources']:
        print(f"- {s['source']} (Page {s['page']}) [Confidence: {s['score']:.2f}]")