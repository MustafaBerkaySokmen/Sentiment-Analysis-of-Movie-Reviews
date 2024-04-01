import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score


# Preprocessing functions
def clean_text(text):
    text = re.sub(r'[.,()\-]', ' ', text)
    return text.split()

def lower_case(text_list):
    return [word.lower() for word in text_list]

def preprocess_data(reviews):
    cleaned_reviews = []
    for review in reviews:
        clean_data = clean_text(review)
        rejoined_text = ' '.join(lower_case(clean_data))
        cleaned_reviews.append(rejoined_text)
    return cleaned_reviews

# Load and preprocess training data
train_data_path = 'train.csv'
test_data_path = 'test.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    preprocess_data(train_data['review']), train_data['sentiment'], test_size=0.2, random_state=42)

# Tokenization and Padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)

maxlen = 1000
train_padded = pad_sequences(train_sequences, maxlen=maxlen)
val_padded = pad_sequences(val_sequences, maxlen=maxlen)

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=5000, output_dim=64, input_length=maxlen))
lstm_model.add(LSTM(25, dropout=0.3, recurrent_dropout=0.3))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(1, activation='sigmoid'))

# Compile and fit model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
lstm_model.fit(train_padded, train_labels, epochs=10, batch_size=64, 
               validation_data=(val_padded, val_labels), callbacks=[early_stopping])

# Evaluate model
val_predictions = (lstm_model.predict(val_padded) > 0.5).astype("int32")
precision = precision_score(val_labels, val_predictions)
recall = recall_score(val_labels, val_predictions)
f1 = f1_score(val_labels, val_predictions)

print('LSTM Model Evaluation')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Predict sentiments on test data
test_texts = preprocess_data(test_data['review'])
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=maxlen)
predicted_sentiments = (lstm_model.predict(test_padded) > 0.5).astype("int32")

# Create submission file
submission = pd.DataFrame({
    "id": test_data['id'],
    "sentiment": predicted_sentiments.flatten()
})

# Save submission file
submission.to_csv('/kaggle/working/submission.csv', index=False)