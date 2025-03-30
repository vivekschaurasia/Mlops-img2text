from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import build_model
from data_preprocessing import load_captions, get_max_length


app = Flask(__name__)

# Load resources
print("[INFO] Loading model & tokenizer...")
captions = load_captions("Dataset/captions.txt")
features = pickle.load(open("outputs/image_features.pkl", "rb"))
tokenizer = pickle.load(open("outputs/tokenizer.pkl", "rb"))

vocab_size = len(tokenizer.word_index) + 1
max_length = get_max_length(captions)

model = build_model(vocab_size, max_length)
model.load_weights("outputs/model/saved_model_weights.h5")


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(image, tokenizer):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    image = Image.open(file).resize((299, 299))
    image = img_to_array(image)
    image = image.reshape((1, 299, 299, 3))
    image = image / 127.5 - 1  # Preprocess

    # Fake ID for demonstration
    feature = model.get_layer(index=1).input  # skip if using features
    caption = predict_caption(image, tokenizer)

    return jsonify({'caption': caption})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
