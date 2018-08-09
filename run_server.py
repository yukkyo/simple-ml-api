from sklearn.externals import joblib
import flask
import numpy as np

# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)
model = None


def load_model():
    global model
    print(" * Loading pre-trained model ...")
    model = joblib.load("./trained-model/sample-model.pkl")
    print(' * Loading end')


@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json().get("feature"):
            # read feature from json
            feature = flask.request.get_json().get("feature")

            # preprocess for classification
            # list  -> np.ndarray
            feature = np.array(feature).reshape((1, -1))

            # classify the input feature
            response["prediction"] = model.predict(feature).tolist()

            # indicate that the request was a success
            response["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(response)


if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run()
