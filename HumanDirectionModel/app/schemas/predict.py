from pydantic import BaseModel, Field

class PredictResponse(BaseModel):
    label: str = Field(description='One of: Front | Left | Right | Full Body | N/A')
    meta: dict = Field(default_factory=dict)
