 Mini RAG Chatbot:

A beginner-friendly AI chatbot that combines Retrieval-Augmented Generation (RAG) with a Hugging Face language model.

1. Features
- Reads information from a custom knowledge file.
- Generates text embeddings using Sentence Transformers.
- Performs semantic search using cosine similarity.
- Retrieves relevant context for user questions.
- Uses a Hugging Face LLM for text generation.

2. Technologies Used
- Python
- Sentence Transformers
- Hugging Face Transformers
- RAG
- Semantic Search
- Cosine Similarity

3. Project Structure
Mini-RAG-Bot/
- app.py
- knowledge.txt
- README.md

4. How to Run

Install dependencies:

```bash
pip install sentence-transformers transformers torch
```

Run the chatbot:

```bash
python app.py
```

5. Learning Outcomes
Learned the fundamentals of RAG, embeddings, semantic search, document retrieval and LLM integration.
