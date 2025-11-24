import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model():
    # create sample dataset
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2.2, 4.3, 6.0, 8.1, 10.2]
    })

    X = data[['x']]
    y = data['y']

    # train model
    model = LinearRegression()
    model.fit(X, y)

    # save model
    save_path = os.path.join(os.getcwd(), "model.joblib")
    joblib.dump(model, save_path)

    print(f"Model saved at: {save_path}")

if __name__ == "__main__":
    train_model()
