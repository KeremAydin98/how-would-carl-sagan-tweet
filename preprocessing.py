import tensorflow as tf
import numpy as np
import config

def load_data(train_data_path):

    # Load the text file
    with open(train_data_path,"r") as f:
        text = f.read()

    # Initialize the char tokenizer
    char_tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, char_level=True)

    # Tokenize our training data
    char_tokenizer.fit_on_texts(text)

    # Encode train sentences into sequences
    [train_sequences] = np.array(char_tokenizer.texts_to_sequences([text])) - 1

    return char_tokenizer, train_sequences

def window_data(dataset,window_size=101):

    # Turn dataset into tf.data.Dataset format
    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # Window the dataset
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Batch the dataset
    dataset = dataset.flat_map(lambda x: x.batch(window_size))

    # Shuffle the batches
    dataset = dataset.shuffle(10000).batch(8,drop_remainder=True)

    # Split the data into features and target
    dataset = dataset.map(lambda windows: (windows[:,:-1], windows[:,1:]))

    # One hot encoding the features
    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=config.max_id), Y_batch))

    return dataset


