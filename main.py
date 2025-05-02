import joblib

# Load the model
model = joblib.load('language_detection_model.pkl')

# Function to predict language
def detect_language(text):
    if not text.strip():
        return "⚠️ Empty input"
    try:
        prediction = model.predict([text])[0]
        return f"🔤 Predicted language: {prediction}"
    except Exception as e:
        return "❌ Error during prediction"

# Interactive input loop
while True:
    user_input = input("📝 Enter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("👋 Exiting.")
        break
    result = detect_language(user_input)
    print(result)
