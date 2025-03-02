import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load all datasets
def load_data(filepaths):
    data = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as file:
            data.extend(json.load(file))
    return data

# Define dataset file paths
dataset_files = [
    'E:\datasets\Chatbot_Artists.json',
    'E:\datasets\Chatbot_Artworks.json',
    'E:\datasets\Chatbot_Art_Epochs.json',
    'E:\datasets\Chatbot_General_Conversation.json'
]

data = load_data(dataset_files)

# Extract questions and answers
questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1

question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Padding
max_length = max(len(seq) for seq in question_sequences + answer_sequences)
question_padded = pad_sequences(question_sequences, maxlen=max_length, padding='post')
answer_padded = pad_sequences(answer_sequences, maxlen=max_length, padding='post')

# Reshape answer labels to match sequence output
y_train = np.expand_dims(answer_padded, axis=-1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(question_padded, y_train, test_size=0.2, random_state=42)

# Build LSTM Model
def build_model(vocab_size, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model

# Initialize model
model = build_model(vocab_size, max_length)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Save the trained model
model.save('E:\datasets\chatbot_model.h5')

# Save tokenizer
with open('E:\datasets\tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer.word_index, f)

print("Model training complete! The model and tokenizer are saved in E:\datasets\")





