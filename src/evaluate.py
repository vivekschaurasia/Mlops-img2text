import os
import pickle
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mlflow

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


def evaluate_model(model, test_keys, features, captions, tokenizer, max_length):
    actual, predicted = [], []
    print("[INFO] Generating predictions...")
    for key in test_keys:
        actual_captions = [cap.split() for cap in captions[key]]
        y_pred = predict_caption(model, features[key], tokenizer, max_length).split()

        actual.append(actual_captions)
        predicted.append(y_pred)
    print("[INFO] Done.")
    return actual, predicted


if __name__ == "__main__":
    # === Paths ===
    captions_path = "../Dataset/captions.txt"
    features_path = "../outputs/image_features.pkl"
    tokenizer_path = "../outputs/tokenizer.pkl"

    # === Load Data ===
    print("[INFO] Loading data...")
    captions = load_captions(captions_path)
    features = pickle.load(open(features_path, 'rb'))
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))

    vocab_size = len(tokenizer.word_index) + 1
    max_length = get_max_length(captions)

    keys = list(captions.keys())
    test_keys = keys[int(0.9 * len(keys)):]

    # === Load Model ===
    print("[INFO] Building model...")
    model = build_model(vocab_size, max_length)
    model.load_weights("..\outputs\model\saved_model_weights.h5")  # Replace if saved differently

    # === Evaluate ===
    actual, predicted = evaluate_model(model, test_keys, features, captions, tokenizer, max_length)

    # === Compute BLEU Scores ===
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    print(f"[RESULT] BLEU-1: {bleu1:.4f}")
    print(f"[RESULT] BLEU-2: {bleu2:.4f}")
    print(f"[RESULT] BLEU-3: {bleu3:.4f}")
    print(f"[RESULT] BLEU-4: {bleu4:.4f}")

    # === Log with MLflow ===
    with mlflow.start_run(run_name="eval-captioning"):
        mlflow.log_metric("BLEU-1", bleu1)
        mlflow.log_metric("BLEU-2", bleu2)
        mlflow.log_metric("BLEU-3", bleu3)
        mlflow.log_metric("BLEU-4", bleu4)

    print("[INFO] BLEU scores logged to MLflow.")
