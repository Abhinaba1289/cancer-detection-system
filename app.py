from flask import Flask, request, jsonify, render_template
import h2o
import pandas as pd

app = Flask(__name__)
h2o.init()

model = h2o.load_model("models\GLM_1_AutoML_2_20250609_115421")
  # change this to your model name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-single', methods=['POST'])
def predict_single():
    data = request.json
    df = pd.DataFrame([data])
    h2o_df = h2o.H2OFrame(df)
    pred = model.predict(h2o_df).as_data_frame()

    label = pred['predict'][0]
    confidence = round(pred['M'][0] * 100, 2)
    return jsonify({'label': label.capitalize(), 'confidence': confidence})

@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    file = request.files['file']
    df = pd.read_csv(file)
    h2o_df = h2o.H2OFrame(df)
    preds = model.predict(h2o_df).as_data_frame()

    results = []
    for _, row in preds.iterrows():
        label = row['predict'].capitalize()
        confidence = round(row['M'] * 100, 2)
        results.append({'label': label, 'confidence': confidence})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
