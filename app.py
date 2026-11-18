
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load knowledge
with open("knowledge.txt", "r", encoding="utf-8") as file:
    documents = [
        line.strip()
        for line in file
        if line.strip()
    ]

# 2. Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Generate document embeddings
document_embeddings = embedding_model.encode(
    documents,
    convert_to_tensor=True
)

# 4. Load LLM
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name)

print("Mini RAG Chatbot Ready!")
print("Type exit to stop.")

# 5. Chat loop
while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    # 6. Question embedding
    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )

    # 7. Find similarity
    scores = cos_sim(
        question_embedding,
        document_embeddings
    )[0]

    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    # 8. Retrieve context
    if best_score < 0.40:
        print("Bot: I couldn't find relevant information.")
        continue

    context = documents[best_index]

    # 9. Create prompt
    prompt = (
        f"Information: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )

    # 10. Generate answer using LLM
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = llm.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print("Bot:", answer)
