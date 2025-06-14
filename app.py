from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
pipe = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    output = pipe("summarize: " + text, max_length=50)
    return jsonify(output)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
