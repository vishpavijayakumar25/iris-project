from fastapi import FastAPI
import joblib
app=FastAPI()
model=joblib.load("iris_model.pkl")
flowers=["setosa","versicolor","virginica"]
@app.get("/")
def home():
         return{"message" : "Iris prediction API"}
@app.get("/predict")
def predict():
         result=model.predict([[5.1,3.5,1.11,0.2]])
         return {
          "prediction":flowers[result[0]]
         }