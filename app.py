from flask import Flask, request, jsonify
from transformers import pipeline, T5Tokenizer

app = Flask(__name__)
pipe = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
tokenizer = pipe.tokenizer

def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    text_chunks = chunk_text(text)
    result = []
    
    for chunk in text_chunks:
        mxl = int(len(chunk) * 0.40)
        mnl = int(len(chunk) * 0.25)
        output = pipe( chunk, max_length=mxl, min_length=mnl, num_beams=4, early_stopping=True)
        result.append(output[0]["generated_text"])

    return jsonify({"summary": " ".join(result)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
