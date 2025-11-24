import joblib
import os

def evaluate_model():
    # Correct path to saved model
    model_path = os.path.join(os.getcwd(), "model.joblib")

    # Load model
    model = joblib.load(model_path)

    # Make a test prediction
    prediction = model.predict([[6]])
    print("Prediction for x=6:", prediction)

if __name__ == "__main__":
    evaluate_model()
