from pydantic import BaseModel


class WeatherReport(BaseModel):
    country: str
    city: str
    temperature: float


class WeatherQuery(BaseModel):
    country: str
    city: str


class WeatherResult(BaseModel):
    temperature_c: float
