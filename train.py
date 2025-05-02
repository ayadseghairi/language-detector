import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv("cleaned_balanced_dataset.csv")  

# Prepare features and target
X = df['Text']
y = df['Language']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with KNN
model = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),  
    ('clf', KNeighborsClassifier(n_neighbors=5))  # Using KNN with 5 neighbors
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'language_detection_model.pkl')
print("KNN Model saved as language_detection_model.pkl")