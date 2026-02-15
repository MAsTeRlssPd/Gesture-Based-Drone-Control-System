import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
data = pd.read_csv('drone_dataset.csv')
data['label'] = data['label'].str.upper()
X = data.drop('label', axis=1)
y = data['label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=40)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)), 
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])   
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Success! 'gesture_model.pkl' created.")
print("Training Simple ANN...")
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2)
preds = np.argmax(model.predict(X_test_scaled), axis=1)
acc = accuracy_score(y_test, preds)
print(f"Overall Accuracy: {acc * 100:.2f}%")
