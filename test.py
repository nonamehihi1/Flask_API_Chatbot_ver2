import pandas as pd
import numpy as np
import random
from ctransformers import AutoModelForCausalLM
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)
CORS(app)

# model embedding từ GPT4ALL
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embedding_model = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
)

# Đọc file CSV chứa mô tả con mèo
try:
    df = pd.read_csv('Description of con mèo.csv')
    contexts = df['Description'].tolist()
    print("CSV file loaded successfully")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    contexts = []

# Khởi tạo mô hình ngôn ngữ
model_path = "vinallama-7b-chat_q5_0.gguf"

try:
    llm = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama")
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    llm = None

# Chia nhỏ văn bản
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=512,
    chunk_overlap=20,
    length_function=len
)

# Vector hóa contexts
chunks = []
for context in contexts:
    chunks.extend(text_splitter.split_text(context))

db = FAISS.from_texts(texts=chunks, embedding=embedding_model)

def find_relevant_contexts(question, top_k=3):
    docs = db.similarity_search(question, k=top_k)
    return [doc.page_content for doc in docs]

def generate_creative_prompt(question, relevant_contexts):
    context_str = ' '.join(relevant_contexts)
    return f"Dựa vào thông tin sau: '{context_str}', hãy trả lời ngắn gọn câu hỏi: {question}"

def generate_answer(question, max_length=70):
    if llm is None:
        return "Error: Model not loaded"

    relevant_contexts = find_relevant_contexts(question)
    creative_prompt = generate_creative_prompt(question, relevant_contexts)

    response = llm(creative_prompt, max_new_tokens=max_length, temperature=0.5, top_p=0.7)

    # Cắt bớt câu trả lời nếu vẫn dài hơn max_length
    truncated_response = response[:max_length]
    last_sentence_end = max(
        truncated_response.rfind('.'),
        truncated_response.rfind('!'),
        truncated_response.rfind('?')
    )
    if last_sentence_end > 0:
        truncated_response = truncated_response[:last_sentence_end + 1]

    return truncated_response

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.json.get('question')
    generated_answer = generate_answer(user_input)
    response = {
        'generated_answer': generated_answer
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)