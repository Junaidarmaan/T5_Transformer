from flask import Flask, request, jsonify
from transformers import pipeline, T5Tokenizer
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
import nltk
import os

nltk.download("punkt")

app = Flask(__name__)
pipe = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
tokenizer = pipe.tokenizer

def smart_chunk_text(text, max_tokens=512):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        combined = current_chunk + " " + sentence if current_chunk else sentence
        if len(tokenizer.encode(combined)) <= max_tokens:
            current_chunk = combined
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def remove_redundant(sentences, threshold=0.85):
    filtered = []
    for sent in sentences:
        if all(SequenceMatcher(None, sent, prev).ratio() < threshold for prev in filtered):
            filtered.append(sent)
    return filtered

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    text_chunks = smart_chunk_text(text)
    results = []

    for chunk in text_chunks:
        max_len = int(len(chunk) * 0.4)
        min_len = int(len(chunk) * 0.25)
        prompt = "Summarize this paragraph: " + chunk
        output = pipe(prompt, max_length=max_len, min_length=min_len, num_beams=4, early_stopping=True)
        results.append(output[0]["generated_text"].strip())

    unique_summaries = remove_redundant(results)
    final_summary = " ".join(unique_summaries)
    return jsonify({"summary": final_summary})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
