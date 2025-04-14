import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample dummy dataset
data = {
    'title': [
        'AI in Marketing', 'History of Rome', 'Meditation Guide',
        'Deep Learning Basics', 'Startup Funding Tips'
    ],
    'description': [
        'Explore how AI is changing the world of marketing.',
        'A deep dive into the Roman empire and its emperors.',
        'A step-by-step guide to mindfulness and inner peace.',
        'Basics of deep learning and neural networks.',
        'Learn how to secure funding for your startup journey.'
    ],
    'duration_minutes': [22, 45, 15, 30, 18]  # Target
}

df = pd.DataFrame(data)
X_text = df['title'] + " " + df['description']
y = df['duration_minutes']

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluation
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"Test RMSE: {rmse:.2f}")

# Predict new sample
sample = ["The Future of AI", "Insights into upcoming AI tools and trends."]
sample_vec = vectorizer.transform([" ".join(sample)])
predicted_duration = model.predict(sample_vec)[0]
print(f"Predicted duration: {predicted_duration:.2f} minutes")
