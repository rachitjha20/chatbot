import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load intents JSON file
intents = json.loads(open('intent.json').read())

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained model
model = load_model('chatbot_model.h5')

# Preprocess input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return the intent with probability
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_list

# Get response based on predicted intent
# Get response based on predicted intent
def get_response(intent_list, intent_json):
    tag = intent_list[0]['intent']
    list_of_intents = intent_json['intents']
    result = "Sorry, I don't understand that."  # Default response
    for i in list_of_intents:
        if i['intent'] == tag:
            result = random.choice(i['responses'])  # Ensure 'responses' key is used
            break
    return result


# Start chatbot
print("Bot is running...")

while True:
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)
