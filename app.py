from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Annotated,Literal
from pydantic import BaseModel, Field, computed_field
import joblib
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


model = joblib.load("Travel_Insurance_Model.pkl")


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

    input_df = pd.DataFrame([{
       "age":insurance.Age, 
       "employment_type":insurance.Employment_type,
       "graduateornot":insurance.graduated, 
       "chronicdiseases":insurance.cronic_disease,
       "frequentflyer":insurance.frequent_flyer, 
       "evertravelledabroad":insurance.ever_travelled_abroad,
       "income":insurance.income, 
       "family":insurance.family
    }]
    )

    result=int(model.predict(input_df)[0])

    return JSONResponse(status_code=200,content={"predicted":result})

