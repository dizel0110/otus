import pickle
from sklearn.linear_model import LogisticRegression
from flask import Flask
from flask_restx import Api, Resource, fields
#from werkzeug.contrib.fixers import ProxyFix
#from werkzeug.middleware.proxy_fix import ProxyFix

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)
#app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='ML API Heart', validate=True)

ns = api.namespace('heart', description='HD model')

heart_row = api.model('Heart_Desease', {
    'age': fields.Float(required=True),
    'sex': fields.Float(required=True),
    'cp': fields.Float(required=True),
    'trestbps': fields.Float(required=True),
    'chol': fields.Float(required=True),
    'fbs': fields.Float(required=True),
    'restecg': fields.Float(required=True),
    'thalach': fields.Float(required=True),
    'exang': fields.Float(required=True),
    'oldpeak': fields.Float(required=True),
    'slope': fields.Float(required=True),
    'ca': fields.Float(required=True),
    'thal': fields.Float(required=True)
})

heart_prediction = api.inherit('HeartPrediction', heart_row, {
    'prediction': fields.List(fields.Float, min_items=1, max_items=1)
})


@ns.route('/')
class HeartClassification(Resource):
    @ns.doc('obtain_prediction')
    @ns.expect(heart_row)
    @ns.marshal_with(heart_prediction, code=200)
    def post(self):
        payload = api.payload
        values_tuple = tuple(payload.values())
        prediction = [round(p, 5) for p in model.predict_proba([values_tuple])[0]]
        payload.update({'prediction': prediction})
        return payload


if __name__ == '__main__':
    app.run() #debug=True