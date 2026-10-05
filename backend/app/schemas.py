from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: str = Field(min_length=1, max_length=100)
    area: float = Field(gt=0, le=1000000, allow_inf_nan=False)
    bedrooms: int = Field(ge=1, le=30)
    bathrooms: float = Field(gt=0, le=30, allow_inf_nan=False)
    property_type: str = Field(min_length=1, max_length=100)
    parking: str = Field(min_length=1, max_length=50)
    property_age: int = Field(ge=0, le=300)
    amenities: list[str] = Field(max_length=8)

    @field_validator("location", "property_type", "parking")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank")
        return value

    @field_validator("amenities")
    @classmethod
    def validate_amenities_shape(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Amenities must not contain duplicates")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Amenities must be non-empty strings")
        return [value.strip() for value in values]


class PredictionResponse(BaseModel):
    estimated_price: int
    formatted_price: str
