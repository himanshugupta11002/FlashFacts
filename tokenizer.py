import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Load the datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Select only 'text' and 'title' columns
fake_df = fake_df[['text', 'title']]
true_df = true_df[['text', 'title']]

# Concatenate the datasets
data = pd.concat([fake_df, true_df], ignore_index=True)

# Concatenate text columns
data['all_text'] = data['title'] + ' ' + data['text']

# Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['all_text'])

# Save the tokenizer's word index
np.save('tokenizer_word_index.npy', tokenizer.word_index)
