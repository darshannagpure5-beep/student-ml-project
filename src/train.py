import joblib
from utils import load_data
from model import get_model

data = load_data("data/student_data.csv")

X = data.drop("pass", axis=1)
y = data["pass"]

model = get_model()
model.fit(X, y)

joblib.dump(model, "models/model.pkl")

print("Model trained and saved!")
