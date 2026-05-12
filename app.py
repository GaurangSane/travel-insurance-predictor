from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Annotated,Literal
from pydantic import BaseModel, Field, computed_field
import pandas as pd
import traceback

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


model = None


import os
import joblib

BASE_DIR = os.path.dirname(__file__)
print(os.listdir(BASE_DIR))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Travel_Insurance_Model.pkl"
)

print("MODEL PATH:", MODEL_PATH)

model = joblib.load(MODEL_PATH)

print("MODEL LOADED SUCCESSFULLY")


class Insurance_Input(BaseModel):
    Age : Annotated[int,Field(...,gt=0,lt=120,description="Age of traveller")]
    Employment_type:Annotated[Literal['Government Sector', 'Private Sector/Self Employed'],Field(...,description="traveller employment type")]
    graduated:Annotated[Literal["Yes","No"],Field(...,description="traveller is graduated?")]
    cronic_disease:Annotated[Literal["Yes","No"],Field(...,description="does traveller has any disease")]
    frequent_flyer:Annotated[Literal["Yes","No"],Field(...,description="do traveller fly frequently")]
    ever_travelled_abroad:Annotated[Literal["Yes","No"],Field(...,description="have you travelled abroad")]
    annual_income:Annotated[float,Field(...,gt=0,description="annual income of traveller")]
    family_members:Annotated[int,Field(...,description='number of people in family')]

    

    @computed_field
    @property
    def income(self) -> str:
        if self.annual_income<500000:
            return "low"
        if self.annual_income<1000000:
            return "medium"
        if self.annual_income<1500000:
            return "high"
        else:
            return "very high"
        
    @computed_field
    @property
    def family(self) -> str:
        if self.family_members<3:
            return "small"
        if self.family_members<6:
            return "medium"
        else:
            return "big"  


@app.get("/")
def home():

    return {
        "message": "Travel Insurance API Running"
    }          
        

@app.post("/predict")
def predict_insurance(insurance : Insurance_Input):
    try:

        input_df = pd.DataFrame([{

            "Age": insurance.Age,

            "Employment_type":
                insurance.Employment_type,

            "graduated":
                insurance.graduated,

            "cronic_disease":
                insurance.cronic_disease,

            "frequent_flyer":
                insurance.frequent_flyer,

            "ever_travelled_abroad":
                insurance.ever_travelled_abroad,

            "annual_income":
                insurance.annual_income,

            "family_members":
                insurance.family_members
        }])

        prediction = model.predict(input_df)[0]

        return {
            "predicted": int(prediction)
        }

    except Exception as e:

        error_trace = traceback.format_exc()

        return JSONResponse(
            status_code=500,

            content={
                "error": str(e),
                "traceback": error_trace
            }
        )

