from pydantic import BaseModel

# create a class with pydantic that will help structuring input data
class credit_application(BaseModel):
    variable : float
