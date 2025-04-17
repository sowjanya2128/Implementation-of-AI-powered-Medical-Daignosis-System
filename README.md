import pandas as pd from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestClassifier from sklearn.metrics import accuracy_score

Load dataset
medical_data = pd.read_csv("medical_dataset.csv") X = medical_data.drop(columns=['disease']) # Symptoms as input Y = medical_data['disease'] # Target labels

Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

Train model
model = RandomForestClassifier() model.fit(X_train, Y_train)

Evaluate model
predictions = model.predict(X_test) print("Accuracy:", accuracy_score(Y_test, predictions)) from flask import Flask, request, jsonify import pickle

app = Flask(name)

Load trained model
model = pickle.load(open("medical_model.pkl", "rb"))

@app.route('/predict', methods=['POST']) def predict(): data = request.json['symptoms'] # Expecting JSON input prediction = model.predict([data]) return jsonify({"diagnosis": prediction[0]})

if name == 'main': app.run(debug=True) from flask import Flask, request, jsonify import pickle

app = Flask(name)

Load trained model
model = pickle.load(open("medical_model.pkl", "rb"))

@app.route('/predict', methods=['POST']) def predict(): data = request.json['symptoms'] # Expecting JSON input prediction = model.predict([data]) return jsonify({"diagnosis": prediction[0]})

if name == 'main': app.run(debug=True)
