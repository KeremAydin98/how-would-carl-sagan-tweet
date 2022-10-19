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

model.fit(train_dataset.repeat(), epochs=10000, steps_per_epoch = 50)

# saving
with open('./Models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(char_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



