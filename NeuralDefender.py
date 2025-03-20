import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 📌 Initialize Flask App
app = Flask(__name__)

# 📌 Load Malware Dataset (Use a real dataset here)
data = pd.read_csv("malware_dataset.csv")  # Replace with actual dataset

# 📌 Define Features & Labels
X = data.drop(columns=['malware_label'])  # Features
y = data['malware_label']  # Labels (1 = Malware, 0 = Benign)

# 📌 Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Multiple AI Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True)
}

model_accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    joblib.dump(model, f"{name}_model.pkl")
    print(f"🔹 {name} Model Training Completed - Accuracy: {accuracy:.2f}")

# 📌 Generate Feature Importance Graph (Only for Tree-Based Models)
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=models["RandomForest"].feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance in Malware Detection")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("static/feature_importance.png")  # Save the image for Flask UI

# 📌 Generate Confusion Matrix Graph for RandomForest Model
conf_matrix = confusion_matrix(y_test, models["RandomForest"].predict(X_test))
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malware"], yticklabels=["Benign", "Malware"])
plt.title("Confusion Matrix - RandomForest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/confusion_matrix.png")  # Save image for UI

# 📌 Define Features Used in Training
FEATURES = list(X.columns)

# 📌 Home Page Route
@app.route('/')
def home():
    return render_template('index.html', accuracies=model_accuracies)

# 📌 File Upload & Malware Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # Simulated feature extraction (Replace this with actual file analysis)
    file_features = pd.DataFrame([[12345, 7.5, 50]], columns=FEATURES[:3])  # Example file features
    
    # Load Selected Model (Default: RandomForest)
    selected_model = "RandomForest"
    model = joblib.load(f"{selected_model}_model.pkl")
    
    # Predict using AI Model
    prediction = model.predict(file_features)[0]
    result = "⚠️ Malware Detected!" if prediction == 1 else "✅ File is Safe"

    return jsonify({"result": result, "model_used": selected_model})

# 📌 Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
