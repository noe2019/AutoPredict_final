# Bring in lighweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()
class ScoringItem(BaseModel):

    RIAGEND: int #2  
    RIDAGEYR: float #30 
    RACE: int #2    
    COUPLE: int #1 
    SMOKER: int #1
    EDUC: int #1
    COVERED: int #1 
    INSURANCE:int #1
    FAT:int #1
    Abdobesity:int #1
    TOTAL_ACCULTURATION_SCORE:int #2
    HTN: int #2   
with open("best_model.pkl","rb")  as f:
    model = pickle.load(f)



@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction":yhat} 