from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the summarization pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Use FLAN-T5 to summarize
    prompt = "summarize: " + text
    output = pipe(prompt, max_length=200, min_length=50, num_beams=4, early_stopping=True)

    return jsonify({"summary": output[0]["generated_text"]})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
