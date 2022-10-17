from models import *
from preprocessing import *
import config
import pickle

char_tokenizer, dataset = load_data(config.train_path)

max_id = len(char_tokenizer.word_index)

config.max_id = max_id

train_dataset = window_data(dataset,window_size=101)

# Load the model
model = GRU_model()

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(".Models/best_model.h5", monitor="loss",save_best_only = True, save_weights_only=True)

model.fit(train_dataset, epochs=20, steps_per_epoch = 2000)

# saving
with open('./Models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(char_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



