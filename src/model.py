from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
from tensorflow.keras.activations import relu, softmax


def build_model(vocab_size, max_length):
    # Image feature input (2048-dim vector from Xception)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.2)(inputs1)
    fe2 = Dense(1024, activation=relu)(fe1)
    fe3 = Dense(512, activation=relu)(fe2)
    fe4 = Dense(256, activation=relu)(fe3)
    fe5 = Dense(128, activation=relu)(fe4)

    # Text input (sequence of word tokens)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.2)(se1)
    se3 = LSTM(128)(se2)

    # Decoder (combine image + sequence)
    decoder1 = add([fe5, se3])
    decoder2 = Dense(128, activation=relu)(decoder1)
    outputs = Dense(vocab_size, activation=softmax)(decoder2)

    # Define and compile the model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


# Optional: summary if running directly
if __name__ == "__main__":
    dummy_vocab_size = 8500
    dummy_max_length = 100
    model = build_model(dummy_vocab_size, dummy_max_length)
    model.summary()
