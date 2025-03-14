from transformers import T5ForConditionalGeneration, T5Tokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)
model = T5ForConditionalGeneration.from_pretrained("models/final_model")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def predict(text):
    input_text = "sentiment: " + text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(inputs["input_ids"], max_length=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    text = data.get("text", "")
    prediction = predict(text)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)