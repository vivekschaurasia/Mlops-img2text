import pickle

with open("C:/Users/vivek/OneDrive/Desktop/MLOps image2text/outputs/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    print(type(tokenizer))
