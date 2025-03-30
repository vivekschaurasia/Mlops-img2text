import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_preprocessing import load_captions, get_max_length
from model import build_model


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
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


def display_caption(image_id, image_dir, captions, features, model, tokenizer, max_length):
    img_path = os.path.join(image_dir, f"{image_id}.jpg")

    if not os.path.exists(img_path):
        print(f"[ERROR] Image '{image_id}.jpg' not found!")
        return

    image = Image.open(img_path)
    image_np = features[image_id]

    print("📌 Actual Captions:")
    for cap in captions[image_id]:
        print(f"- {cap}")

    pred_caption = predict_caption(model, image_np, tokenizer, max_length)
    print("\n🤖 Predicted Caption:")
    print(f"> {pred_caption}")

    plt.imshow(image)
    plt.axis('off')
    plt.title(pred_caption.replace("startseq", "").replace("endseq", "").strip())
    plt.show()

    # BLEU score
    references = [cap.split() for cap in captions[image_id]]
    candidate = pred_caption.split()
    smooth = SmoothingFunction().method1
    score = sentence_bleu(references, candidate, smoothing_function=smooth)
    print(f"\n📈 BLEU Score: {score:.4f}")


if __name__ == "__main__":
    # === Paths ===
    captions_path = "../Dataset/captions.txt"
    features_path = "../outputs/image_features.pkl"
    tokenizer_path = "../outputs/tokenizer.pkl"
    image_dir = "../Dataset/Images"
    model_weights_path = "../outputs/model/saved_model_weights.h5"  # adjust as needed

    # === Load Data ===
    print("[INFO] Loading data...")
    captions = load_captions(captions_path)
    features = pickle.load(open(features_path, 'rb'))
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))

    vocab_size = len(tokenizer.word_index) + 1
    max_length = get_max_length(captions)

    # === Build and Load Model ===
    model = build_model(vocab_size, max_length)
    model.load_weights(model_weights_path)

    # === Predict for a given image ID ===
    image_id = input("Enter image ID (without .jpg): ").strip()
    display_caption(image_id, image_dir, captions, features, model, tokenizer, max_length)
