#Import Flask and pickle
from flask import Flask
from flask import request
from flask import jsonify
import pickle

# Loading model and DictVectorizer
# Loading model and dict vector
file_dv, dv = 'dv.bin', None
with open(file_dv,'rb') as f:
    dv = pickle.load(f)

file_model, model1 = 'model1.bin', None
with open(file_model,'rb') as f:
    model1 = pickle.load(f)

# Creation of web service
app = Flask('ml')

@app.route('/predict', methods=['POST'])
def predict():
    x = request.get_json()
    x_sample = dv.transform([x])
    y_pred = model1.predict_proba(x_sample)[0,1]
    result = {'prob_get_credit':float(y_pred)}
    return jsonify(result)
    

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)