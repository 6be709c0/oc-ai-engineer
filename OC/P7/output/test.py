import pickle
import json
import numpy as np
import keras

from keras.models import load_model  

model = load_model('./model.keras')  

with open("./tokenizer.pkl", "rb") as file:  
    vector_model = pickle.load(file)

params = json.load(open("./params.json", 'r', encoding='utf-8'))
       
def pad_sequences(sequence, maxlen, value=0):  
    return np.array([np.pad(s[:maxlen], (max(0, maxlen-len(s)), 0), 'constant', constant_values=value) if len(s) < maxlen else s[:maxlen] for s in sequence])  
  
def predict(comment, model, vector_model):
    # We need to pad sequences to ensure uniform input size  
    sequence = vector_model.texts_to_sequences(["this is a text that is quite long"])
    padded_sequence = pad_sequences(sequence, maxlen=int(params["input_length"]))

    prediction = model.predict(padded_sequence)
    # print(prediction)
    sentiment = prediction.astype(float)[0][0]
    return sentiment.item()
    
def comment_to_vec(comment, model):
    vec = np.zeros(100)
    num_words = 0
    for word in comment:
        if word in model.wv:
            vec += model.wv[word]
            num_words += 1
    if num_words > 0:
        vec /= num_words
    return vec

print(predict("I am so sad, this is very bad news, terrible!", model, vector_model))
print(predict("I am so happy, this is very good news, congrats!", model, vector_model))
print(predict("This is not good", model, vector_model))