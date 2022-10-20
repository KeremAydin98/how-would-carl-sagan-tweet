import string

import numpy as np
import tensorflow as tf
import config
import pickle
from keybert import KeyBERT
import random

def predict_text(model, text, tokenizer, temperature=1):
    x = np.array(tokenizer.texts_to_sequences([text])) - 1

    preprocessed = tf.one_hot(x, config.max_id)

    predicted = model.predict(preprocessed)[0, -1:, :]

    rescaled_logits = tf.math.log(predicted) / temperature

    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1

    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def generate_text(model, tokenizer):

  text = random.choice(string.ascii_lowercase)

  for _ in range(280):

    text += predict_text(model, text, tokenizer)

  return text


model = tf.keras.models.load_model("./Models/best_model.h5")

# loading
with open('./Models/tokenizer.pickle', 'rb') as handle:
    char_tokenizer = pickle.load(handle)

for i in range(5):

    # Generate the tweet of Carl Sagan
    generated_text = generate_text(model, char_tokenizer)

    # Bert Keyphrase Extraction method
    kw_model = KeyBERT()

    # Keyword generation
    keywords = kw_model.extract_keywords(generated_text, keyphrase_ngram_range=(1,1))

    with open(f".Data/Tweet_{i}.txt","w") as f:

        whole_tweet = generated_text + "@" + keywords[0] + "@" + keywords[1]
        f.write(whole_tweet)