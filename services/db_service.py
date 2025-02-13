from pymongo import MongoClient
import os

selected_db_service = None

def set_db_service(service):
    
    global selected_db_service
    selected_db_service = service


class MongoDBService:
    def __init__(self, database_name: str, collection_name: str):
        mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017/")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

    def insert_document(self, document_dict):
        result_insert = self.collection.insert_one(document_dict)
        print("Documento inserido com _id:", result_insert.inserted_id)
        return result_insert.inserted_id