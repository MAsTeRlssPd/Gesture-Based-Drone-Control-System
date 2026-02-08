import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
data = pd.read_csv('drone_dataset.csv')
X = data.drop('label', axis=1)
y = data['label']         
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Training the AI model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Success! 'gesture_model.pkl' created.")
