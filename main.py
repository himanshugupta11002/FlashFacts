import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Load the datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Select only 'text' and 'title' columns
fake_df = fake_df[['text', 'title']]
true_df = true_df[['text', 'title']]

# Add labels to the datasets
fake_df['label'] = 1
true_df['label'] = 0

# Concatenate the datasets
data = pd.concat([fake_df, true_df], ignore_index=True)

# Concatenate text columns
data['all_text'] = data['title'] + ' ' + data['text']

# Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['all_text'])
sequences = tokenizer.texts_to_sequences(data['all_text'])
maxlen = 200  # Reduced maximum length of sequences
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Split the data into features and labels
X = padded_sequences
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define batch size
batch_size = 128

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Calculate number of training and testing batches
num_train_batches = -(-len(X_train) // batch_size)  # Ceiling division
num_test_batches = -(-len(X_test) // batch_size)  # Ceiling division

# Train the model in batches
for epoch in range(5):  # 5 epochs
    print(f"Epoch {epoch + 1}/{5}")
    for batch_index in range(num_train_batches):
        print(f"Training batch {batch_index + 1}/{num_train_batches}")
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(X_train))
        X_batch = X_train[start_index:end_index]
        y_batch = y_train[start_index:end_index]
        model.train_on_batch(X_batch, y_batch)

    # Evaluate on test data after each epoch
    print("Evaluating on test data...")
    total_loss = 0
    total_accuracy = 0
    for batch_index in range(num_test_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(X_test))
        X_batch = X_test[start_index:end_index]
        y_batch = y_test[start_index:end_index]
        loss, accuracy = model.evaluate(X_batch, y_batch, verbose=0)
        total_loss += loss
        total_accuracy += accuracy
    average_loss = total_loss / num_test_batches
    average_accuracy = total_accuracy / num_test_batches
    print(f"Validation loss: {average_loss:.4f}, Validation accuracy: {average_accuracy:.4f}")

# Save the model
model.save('fake_news_detection_model.h5')
