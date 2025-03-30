import os
import pickle
import numpy as np
import mlflow

import mlflow.tensorflow
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical, pad_sequences

from data_preprocessing import load_captions, get_max_length
from model import build_model


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def data_generator(keys, captions, features, tokenizer, max_length, vocab_size, batch_size):
    while True:
        X1, X2, y = list(), list(), list()
        n = 0

        for key in keys:
            n += 1
            desc_list = captions[key]

            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)

            if n == batch_size:
                yield ([np.array(X1), np.array(X2)], np.array(y))
                X1, X2, y = list(), list(), list()
                n = 0


if __name__ == "__main__":
    # === Paths ===
    captions_path = "../Dataset/captions.txt"
    features_path = "../outputs/image_features.pkl"
    tokenizer_path = "../outputs/tokenizer.pkl"
    model_output_dir = "../outputs/model"
    os.makedirs(model_output_dir, exist_ok=True)

    # === Hyperparams ===
    epochs = 3
    batch_size = 32

    # === Load Data ===
    print("[INFO] Loading data...")
    captions = load_captions(captions_path)
    features = load_pickle(features_path)
    tokenizer = load_pickle(tokenizer_path)

    vocab_size = len(tokenizer.word_index) + 1
    max_length = get_max_length(captions)
    train_keys = list(captions.keys())[:int(0.9 * len(captions))]

    # === Build Model ===
    print("[INFO] Building model...")
    model = build_model(vocab_size, max_length)

    steps_per_epoch = len(train_keys) // batch_size

    # === Train with MLflow ===
    with mlflow.start_run(run_name="image-captioning-xception-lstm"):

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("vocab_size", vocab_size)
        mlflow.log_param("max_length", max_length)

        for epoch in range(epochs):
            print(f"[INFO] Starting Epoch {epoch+1}/{epochs}")
            generator = data_generator(train_keys, captions, features, tokenizer, max_length, vocab_size, batch_size)
            model.fit(generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)

        print("[INFO] Logging model...")
        mlflow.tensorflow.log_model(model=model, artifact_path="model")
        
        tokenizer_file = os.path.join(model_output_dir, "tokenizer.pkl")
        with open(tokenizer_file, "wb") as f:
            pickle.dump(tokenizer, f)
        mlflow.log_artifact(tokenizer_file)


        # Save weights for later use
        weights_path = os.path.join(model_output_dir, "saved_model_weights.h5")
        model.save_weights(weights_path)
        print(f"[INFO] Model weights saved to {weights_path}")

        print("[INFO] Training completed and model logged.")
