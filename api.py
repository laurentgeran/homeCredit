import uvicorn 
import pickle
import application
from fastapi import FastAPI



# create the app
app = FastAPI()

# load model
with open(r"clf_feat_over.pkl", "rb") as input_file:
    model = pickle.load(input_file)

@app.get('/')
def home():
    return({'message':'Hello world !'})

@app.post('/predict')
def predict_score(data:application.credit_application):
    data=data.dict()
    var = [data[x] for x in data.keys()]
    score = model.predict(var)
    return(score)

# run the api
if __name__== '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn  api:app --reload


