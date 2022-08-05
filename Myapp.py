import numpy as np
from flask import Flask, request,render_template
import pickle

Myapp= Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@Myapp.route('/')
def home():
    return render_template('index.html')

@Myapp.route('/predict',methods=['POST'])
def predict():

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output =str(prediction)

    return render_template('index.html', prediction_text='Breast Tissue would be $ {}'.format(output))

if __name__ == "__main__":
    Myapp.run(debug=True)