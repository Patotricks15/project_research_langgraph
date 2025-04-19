from pymongo import MongoClient
import os

selected_db_service = None

def set_db_service(service):
    """
    Sets the database service to be used globally.

    Args:
        service (DBService): The database service to be used.
    """
    global selected_db_service
    selected_db_service = service


class MongoDBService:
    def __init__(self, database_name: str, collection_name: str):
        """
        Initialize a MongoDBService instance.

        Args:
            database_name (str): The name of the MongoDB database to connect to.
            collection_name (str): The name of the MongoDB collection to connect to.

        Attributes:
            client (MongoClient): The MongoClient instance used to connect to the MongoDB database.
            db (Database): The MongoDB database instance.
            collection (Collection): The MongoDB collection instance.
        """
        mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27018/")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

    def insert_document(self, document_dict):
        """
        Inserts a document into the MongoDB collection.

        Args:
            document_dict (dict): A dictionary with the fields of the document to be inserted.

        Returns:
            str: The _id of the inserted document.
        """
        result_insert = self.collection.insert_one(document_dict)
        print("Documento inserido com _id:", result_insert.inserted_id)
        return result_insert.inserted_id