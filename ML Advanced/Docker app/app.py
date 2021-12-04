from flask import Flask, jsonify, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

data = pd.read_csv("heart.csv")
target = data ['target']
data=data.drop(['target'], axis=1)


def train_model():
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3)
    dt = DecisionTreeClassifier().fit(train_data, train_target)
    acc = accuracy_score(test_target, dt.predict(test_data))
    return dt, acc


model, accuracy = train_model()

@app.route('/predict', methods=['POST'])
def predict():
    posted_data = request.get_json()
    print(f'posted_data: {posted_data}')
    age = posted_data['age']
    sex = posted_data['sex']
    cp = posted_data['cp']
    trestbps = posted_data['trestbps']
    chol = posted_data['chol']
    fbs = posted_data['fbs']
    restecg = posted_data['restecg']
    thalach = posted_data['thalach']
    exang = posted_data['exang']
    oldpeak = posted_data['oldpeak']
    slope = posted_data['slope']
    ca = posted_data['ca']
    thal = posted_data['thal']
    prediction = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])[0]
    print('target', prediction)

    return jsonify({'target':str(prediction)})


@app.route('/model')
def get_model():
    return jsonify({'name': 'Decision Tree Classifier',
                    'accuracy': accuracy})


if __name__ == '__main__':
    app.run(host='172.17.0.2')
