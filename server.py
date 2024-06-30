from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from waitress import serve

import logging

from model import arbiter_generate

type_map = {
    'short': 150,
    'medium': 250,
    'long': 500
}
PORT = 9000


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


app = Flask(__name__)
CORS(app)
model = load_model("Fardeen7860/President-Text-Speech-Model-gpt2")
tokenizer = load_tokenizer("Fardeen7860/President-Text-Speech-Model-gpt2")


@app.route('/', methods=['POST'])
def handle_request():
    data = request.get_json()

    prompt = data.get('prompt', '')
    prompt_type = data.get('type', '')

    prompt_type = prompt_type.lower()

    max_length = type_map[prompt_type]

    response = arbiter_generate(model, tokenizer, prompt, max_length)
    response = response.replace(" \n", "\n")
    return jsonify({'generated_text': response})


logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=PORT)
