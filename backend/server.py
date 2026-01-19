
# backend/server.py

from fastapi import FastAPI, APIRouter, HTTPException, status
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'prijsvergelijker')]

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Units
UNIT_CONVERSIONS = {"kg":1000, "gr":1, "l":1000, "cl":10, "ml":1, "stuk":1, "stuks":1}
UNIT_GROUPS = {"weight":["kg","gr"], "volume":["l","cl","ml"], "piece":["stuk","stuks"]}

def calculate_unit_price(price: float, quantity: float, unit: str) -> float:
    if price <= 0 or quantity <= 0:
        return 0
    base_quantity = quantity * UNIT_CONVERSIONS.get(unit,1)
    return price / base_quantity

def process_product_prices(product_data: dict) -> dict:
    if 'prices' not in product_data or not product_data['prices']:
        return product_data
    min_unit_price = float('inf')
    cheapest = None
    for price_info in product_data['prices']:
        unit_price = calculate_unit_price(price_info['price'], price_info['quantity'], price_info['unit'])
        price_info['unit_price'] = round(unit_price,4)
        if 0 < unit_price < min_unit_price:
            min_unit_price = unit_price
            cheapest = price_info['store']
    product_data['cheapest_store'] = cheapest
    return product_data

# Models
class StorePrice(BaseModel):
    store: str
    price: float
    quantity: float
    unit: str
    unit_price: float = 0

class Product(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: Optional[str] = None
    prices: List[StorePrice] = []
    cheapest_store: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProductCreate(BaseModel):
    name: str
    category: Optional[str] = None
    prices: List[StorePrice]

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    prices: Optional[List[StorePrice]] = None

class Category(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CategoryCreate(BaseModel):
    name: str

class ShoppingItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    product_id: str
    product_name: str
    selected_store: Optional[str] = None
    checked: bool = False
    quantity: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ShoppingItemCreate(BaseModel):
    product_id: str
    product_name: str
    selected_store: Optional[str] = None
    quantity: int = 1

class ShoppingItemUpdate(BaseModel):
    checked: Optional[bool] = None
    selected_store: Optional[str] = None
    quantity: Optional[int] = None

# Routes
@api_router.get("/")
async def root():
    return {"message":"Prijsvergelijker API"}

@api_router.post("/products", response_model=Product, status_code=status.HTTP_201_CREATED)
async def create_product(input_data: ProductCreate):
    product_dict = input_data.model_dump()
    product_obj = Product(**product_dict)
    doc = product_obj.model_dump()
    doc = process_product_prices(doc)
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    await db.products.insert_one(doc)
    doc['created_at'] = datetime.fromisoformat(doc['created_at'])
    doc['updated_at'] = datetime.fromisoformat(doc['updated_at'])
    return doc

@api_router.get("/products", response_model=List[Product])
async def get_products():
    products = await db.products.find({}, {"_id":0}).to_list(1000)
    for p in products:
        if isinstance(p.get('created_at'), str): p['created_at'] = datetime.fromisoformat(p['created_at'])
        if isinstance(p.get('updated_at'), str): p['updated_at'] = datetime.fromisoformat(p['updated_at'])
    return products

@api_router.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: str):
    product = await db.products.find_one({"id":product_id},{"_id":0})
    if not product: raise HTTPException(status_code=404, detail="Product niet gevonden")
    if isinstance(product.get('created_at'), str): product['created_at'] = datetime.fromisoformat(product['created_at'])
    if isinstance(product.get('updated_at'), str): product['updated_at'] = datetime.fromisoformat(product['updated_at'])
    return product

@api_router.put("/products/{product_id}", response_model=Product)
async def update_product(product_id: str, input_data: ProductUpdate):
    existing = await db.products.find_one({"id":product_id},{"_id":0})
    if not existing: raise HTTPException(status_code=404, detail="Product niet gevonden")
    update_data = input_data.model_dump(exclude_unset=True)
    if update_data:
        update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
        for k,v in update_data.items(): existing[k] = v
        existing = process_product_prices(existing)
        await db.products.update_one({"id":product_id},{"$set":existing})
    return await get_product(product_id)

@api_router.delete("/products/{product_id}")
async def delete_product(product_id: str):
    result = await db.products.delete_one({"id":product_id})
    if result.deleted_count == 0: raise HTTPException(status_code=404, detail="Product niet gevonden")
    await db.shopping_items.delete_many({"product_id":product_id})
    return {"message":"Product verwijderd"}

# Categorie routes
@api_router.post("/categories", response_model=Category, status_code=status.HTTP_201_CREATED)
async def create_category(input_data: CategoryCreate):
    existing = await db.categories.find_one({"name":input_data.name},{"_id":0})
    if existing: raise HTTPException(status_code=400, detail="Categorie bestaat al")
    cat_obj = Category(name=input_data.name)
    doc = cat_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.categories.insert_one(doc)
    doc['created_at'] = datetime.fromisoformat(doc['created_at'])
    return doc

@api_router.get("/categories", response_model=List[Category])
async def get_categories():
    categories = await db.categories.find({}, {"_id":0}).to_list(100)
    for c in categories:
        if isinstance(c.get('created_at'), str): c['created_at'] = datetime.fromisoformat(c['created_at'])
    return categories

@api_router.delete("/categories/{category_id}")
async def delete_category(category_id: str):
    result = await db.categories.delete_one({"id":category_id})
    if result.deleted_count==0: raise HTTPException(status_code=404, detail="Categorie niet gevonden")
    await db.products.update_many({"category":category_id}, {"$set":{"category":None}})
    return {"message":"Categorie verwijderd"}

# Shopping list routes
@api_router.post("/shopping", response_model=ShoppingItem, status_code=status.HTTP_201_CREATED)
async def add_to_shopping_list(input_data: ShoppingItemCreate):
    item_obj = ShoppingItem(**input_data.model_dump())
    doc = item_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.shopping_items.insert_one(doc)
    doc['created_at'] = datetime.fromisoformat(doc['created_at'])
    return doc

@api_router.get("/shopping", response_model=List[ShoppingItem])
async def get_shopping_list():
    items = await db.shopping_items.find({}, {"_id":0}).to_list(1000)
    for i in items:
        if isinstance(i.get('created_at'), str): i['created_at'] = datetime.fromisoformat(i['created_at'])
    return items

@api_router.put("/shopping/{item_id}", response_model=ShoppingItem)
async def update_shopping_item(item_id: str, input_data: ShoppingItemUpdate):
    existing = await db.shopping_items.find_one({"id":item_id},{"_id":0})
    if not existing: raise HTTPException(status_code=404, detail="Item niet gevonden")
    update_data = input_data.model_dump(exclude_unset=True)
    if update_data: await db.shopping_items.update_one({"id":item_id},{"$set":update_data})
    updated = await db.shopping_items.find_one({"id":item_id},{"_id":0})
    if isinstance(updated.get('created_at'), str): updated['created_at'] = datetime.fromisoformat(updated['created_at'])
    return updated

@api_router.delete("/shopping/{item_id}")
async def remove_from_shopping_list(item_id: str):
    result = await db.shopping_items.delete_one({"id":item_id})
    if result.deleted_count==0: raise HTTPException(status_code=404, detail="Item niet gevonden")
    return {"message":"Item verwijderd"}

@api_router.delete("/shopping")
async def clear_shopping_list():
    await db.shopping_items.delete_many({})
    return {"message":"Boodschappenlijst leeggemaakt"}

@api_router.get("/shopping/totals")
async def get_shopping_totals():
    items = await db.shopping_items.find({}, {"_id":0}).to_list(1000)
    products = await db.products.find({}, {"_id":0}).to_list(1000)
    product_map = {p['id']:p for p in products}
    totals = {"colruyt":0,"delhaize":0,"aldi":0}
    for item in items:
        product = product_map.get(item['product_id'])
        if not product: continue
        quantity = item.get('quantity',1)
        for price_info in product.get('prices',[]):
            store = price_info['store']
            if store in totals: totals[store] += price_info['price']*quantity
    for store in totals: totals[store] = round(totals[store],2)
    cheapest = min(totals,key=totals.get) if any(totals.values()) else None
    return {"totals":totals,"cheapest_store":cheapest}

# Setup
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS','*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
