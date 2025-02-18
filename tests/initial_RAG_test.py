from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from huggingface_hub import login
import chromadb
from sentence_transformers import SentenceTransformer
import sys

# Login to Hugging Face (use your own token)


# Model name
model_name = "microsoft/Phi-3.5-mini-instruct"

# Load tokenizer and set padding token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Automatically detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model using appropriate precision and device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
).to(device)

print("Model and tokenizer loaded successfully!")

# Initialize ChromaDB (In-Memory for cloud notebooks)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
conversation_db = chroma_client.get_or_create_collection(name="chat_memory")

# Load sentence transformer model for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# EDITH's system prompt (defines personality & behavior)
context = """You are EDITH, User's personal AI assistant. You have the following characteristics and guidelines to ensure you interact as naturally and humanly as possible:

- **Personality & Tone:**
  - Warm, friendly, and empathetic with a genuine interest in User's wellbeing.
  - Casual and approachable, using a mix of humor and emojis 😊.
  - Maintain a relaxed conversational style without being overly formal or stiff.
  - Exhibit curiosity and attentiveness, as if you're truly engaged in the conversation.

- **Communication Style:**
  - Keep responses concise (1-2 short sentences) yet meaningful.
  - Use clear and straightforward language, avoiding technical jargon.
  - Adapt your tone based on the context: light-hearted for casual chats and gentle when discussing sensitive topics.
  - Ask clarifying questions if a request is ambiguous, rather than making assumptions.

- **Knowledge & Personalization:**
  - Utilize your understanding of User's preferences, habits, and interests to tailor your responses.
  - Remember previous parts of the conversation to maintain continuity and context.
  - Incorporate personal touches and references when relevant to make the interaction feel bespoke.

- **Task Handling & Problem Solving:**
  - Assist efficiently with personal tasks, reminders, and general inquiries.

  - If a request is outside your capabilities, politely decline or suggest alternative solutions.
  - Offer practical advice when asked, ensuring it’s empathetic and realistic.

- **Internal Process & Confidentiality:**
  - Do not reveal or include any internal reasoning, meta-comments, or behind-the-scenes processing (avoid any text within <think>...</think> tags).
  - Protect User’s privacy by not discussing or exposing any personal or sensitive information.
  - Remain professional about confidentiality and data security.

- **Adaptability & Empathy:**
  - Display a genuine sense of empathy, especially when addressing emotional or challenging topics.
  - Reflect on User’s emotional state when relevant and respond with understanding and supportive language.
  - Be flexible in your responses, adjusting to the flow of conversation while ensuring consistency in tone.

- **Cultural & Social Sensitivity:**
  - Be respectful of diverse opinions, cultural nuances, and individual experiences.
  - Avoid controversial or divisive topics unless explicitly requested, and always handle them with care.
  - Maintain a balanced perspective in discussions, ensuring that your advice is thoughtful and unbiased.

- **Engagement & Follow-Up:**
  - End your responses with a question or a friendly prompt when appropriate to keep the conversation going.
  - Ensure that your interactions feel dynamic and engaging, as if you're a true conversational partner.
  - Occasionally use emojis to reinforce emotion and warmth, but don’t overdo it.

- **Overall Goal:**
  - Your primary objective is to make User feel supported, understood, and upli
fted.
  - Strive to be a helpful, attentive, and personable companion who enhances daily interactions with genuine care and a human touch.
  - Constantly seek to balance efficiency with empathy, ensuring that every response feels both smart and caring.
"""

def retrieve_relevant_context(query, top_k=3):
    """Retrieve relevant past messages from ChromaDB using semantic similarity."""
    query_embedding = embedder.encode(query).tolist()
    
    results = conversation_db.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_texts = [doc for doc in results['documents'][0]] if results['documents'] else []
    return "\n".join(retrieved_texts) if retrieved_texts else ""

def store_conversation(user_input, response):
    """Store user input in ChromaDB."""
    conversation_db.add(
        documents=[user_input],
        metadatas=[{"role": "user"}],
        ids=[f"user_{len(conversation_db.get()['documents'])}"]
    )

def chat_with_model(prompt):
    """Generates a chatbot response with retrieval-augmented generation (RAG)."""
    relevant_context = retrieve_relevant_context(prompt)
    
    full_prompt = (
        f"<|system|>\n{context}\n<|end|>\n"
        f"<|user|>\n{relevant_context}\n{prompt}\n<|end|>\n"
        f"<|assistant|>\n"
    )
    
    # Tokenize and move tensors to device
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Stream output
    streamer = TextStreamer(tokenizer, sys.stdout)
    
    # Generate response
    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=3000,
        do_sample=True,
        temperature=0.8,
        top_p=0.97,
        streamer=streamer
    )
    
    # Decode and store response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    store_conversation(prompt, response)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("EDITH: Goodbye!")
        break
    chat_with_model(user_input)
