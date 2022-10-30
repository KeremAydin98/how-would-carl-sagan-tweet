import string
import numpy as np
import tensorflow as tf
import config
import pickle
from keybert import KeyBERT
import random
from textblob import TextBlob


def predict_text(model, text, tokenizer, temperature=1):

    # Convert to text into sequences(assigned numbers)
    x = np.array(tokenizer.texts_to_sequences([text])) - 1

    # One hot encode the sequences with the depth of max id(number of characters)
    preprocessed = tf.one_hot(x, config.max_id)

    # Get the last character to predict the next character
    predicted = model.predict(preprocessed)[0, -1:, :]

    # Take the logarithm of the predicted sequence
    rescaled_logits = tf.math.log(predicted) / temperature

    # Draws one sample from a categorical distribution
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1

    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def generate_text(model, tokenizer):

    # Choose a random lowercase letter
    text = random.choice(string.ascii_lowercase)

    # Produce 279 letters to match the max letter limit of Twitter which is 280 characters
    for _ in range(250):

        text += predict_text(model, text, tokenizer)

    return text


# Load the best model trained on training script
model = tf.keras.models.load_model("./Models/best_model.h5")

# Loading the tokenizer
with open('./Models/tokenizer.pickle', 'rb') as handle:
    char_tokenizer = pickle.load(handle)


# Initialize Bert Keyphrase Extraction method
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')

# Generating tweets and hashtags
for i in range(5):

    # Generate the tweet of Carl Sagan
    generated_text = generate_text(model, char_tokenizer)

    # Spelling correction
    new_doc = TextBlob(generated_text)

    generated_text = str(new_doc.correct())

    # Keyword extraction
    keywords = kw_model.extract_keywords(generated_text)

    # Write the tweets on text files
    with open(f"Data/Tweet_{i}.txt","w") as f:

        whole_tweet = generated_text + "@" + keywords[0][0] + "@" + keywords[1][0]
        whole_tweet = whole_tweet.encode("ascii", "ignore")
        whole_tweet = whole_tweet.decode()
        f.write(whole_tweet)