from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and labels
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("emotion_labels.pkl", "rb") as f:
    emotion_labels = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    prediction = []
    top_emotion = None

    if request.method == "POST":
        text_input = request.form["text"]

        # Predict probabilities
        y_proba = model.predict_proba([text_input])
        threshold = 0.10

        # Get emotions above threshold
        predicted_emotions = [emotion_labels[i] for i, prob in enumerate(y_proba[0]) if prob > threshold]

        # If none pass threshold, return top-1 emotion anyway
        if not predicted_emotions:
            top_index = np.argmax(y_proba[0])
            predicted_emotions = [emotion_labels[top_index] + " (weak confidence)"]

        prediction = predicted_emotions

    return render_template("index.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
