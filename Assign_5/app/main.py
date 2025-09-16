from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

class Item(BaseModel):
    name: str
    price: float
    tax: float = 0.0

@app.post("/items/")
async def create_item(item: Item):
    total = item.price + item.tax
    return {"name": item.name, "total_price": total}
