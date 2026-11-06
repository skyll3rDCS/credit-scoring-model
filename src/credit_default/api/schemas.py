"""Контракт JSON для /predict"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    LIMIT_BAL: float = Field(..., examples=[20000])
    SEX: int = Field(..., examples=[2])
    EDUCATION: int = Field(..., examples=[2])
    MARRIAGE: int = Field(..., examples=[1])
    AGE: int = Field(..., examples=[24])
    PAY_0: int = Field(..., examples=[2])
    PAY_2: int = Field(..., examples=[2])
    PAY_3: int = Field(..., examples=[-1])
    PAY_4: int = Field(..., examples=[-1])
    PAY_5: int = Field(..., examples=[-2])
    PAY_6: int = Field(..., examples=[-2])
    BILL_AMT1: float = Field(..., examples=[3913])
    BILL_AMT2: float = Field(..., examples=[3102])
    BILL_AMT3: float = Field(..., examples=[689])
    BILL_AMT4: float = Field(..., examples=[0])
    BILL_AMT5: float = Field(..., examples=[0])
    BILL_AMT6: float = Field(..., examples=[0])
    PAY_AMT1: float = Field(..., examples=[0])
    PAY_AMT2: float = Field(..., examples=[689])
    PAY_AMT3: float = Field(..., examples=[0])
    PAY_AMT4: float = Field(..., examples=[0])
    PAY_AMT5: float = Field(..., examples=[0])
    PAY_AMT6: float = Field(..., examples=[0])


class Prediction(BaseModel):
    prediction: int = Field(..., description="1 - дефолт в следующем месяце, 0 - нет")
    probability: float = Field(..., description="Оценка P(default)")


class BatchRequest(BaseModel):
    records: list[ClientFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


class Health(BaseModel):
    status: str
    model_loaded: bool
