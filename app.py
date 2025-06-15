from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the summarization pipeline and tokenizer
pipe = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
tokenizer = pipe.tokenizer

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Encode the text to tokens
    tokens = tokenizer.encode(text, truncation=True, return_tensors="pt")[0]

    # Limit to first 500 tokens if necessary
    if len(tokens) > 500:
        tokens = tokens[:500]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    mxl = int(len(tokens) * 0.35)
    mnl = int(len(tokens) * 0.25)
    # Summarize
    prompt = "summarize the following passage in brief: " + text
    output = pipe(
        prompt, 
        max_length=mxl, 
        min_length=mnl,
        top_p=0.95,
        temperature=0.7,
        num_beams=3, 
        early_stopping=True)

    return jsonify({"summary": output[0]["generated_text"]})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
