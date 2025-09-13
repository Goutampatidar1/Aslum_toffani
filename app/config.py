from pymongo import MongoClient , errors
import os
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL" , None)
DATABASE_NAME = os.getenv("DATABASE_NAME" , "my-database")

print(f'DATABASE NAME : {DATABASE_NAME} , DATABASE_URL : {DATABASE_URL}')

try:
    client = MongoClient(DATABASE_URL , serverSelectionTimeoutMS = 5000)
    client.server_info()
    db  = [DATABASE_NAME]
    print("DATABSE CONNNECTED SUCCESSFULLY")
except errors.ServerSelectionTimeoutError as ex:
    print('Failed to Connect to database')
    raise SystemExit(1)