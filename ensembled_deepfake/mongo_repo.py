from pymongo import MongoClient


class MongoRepository:
    def __init__(self, uri: str = "mongodb://localhost:27017/", db_name: str = "TFGmongodb", collection: str = "Videos"):
        self._client = MongoClient(uri)
        self._db = self._client[db_name]
        self._collection = self._db[collection]


    def insert_one(self, query: dict):
        
        return self._collection.update_one({"video_id" : query["video_id"]}, {"$set": query}, upsert=True)


    def update_videoid(self, video_id: str, changes: dict):
        return self._collection.update_one(
            {"video_id": video_id},     
            {"$set": changes}        
        )


    def update_video_path(self, video_path: str, changes: dict):
        return self._collection.update_one(
            {"video_path": video_path},     
            {"$set": changes}        
        )


    def get_video_path(self, video_path: str):
        return self._collection.find({"video_path": video_path})
    

    def get_videoid(self, video_id: str):
        return self._collection.find({"video_id": video_id})


    def get_from_prompt(self, query=None):
        if query == None:
            query = {}
        return self._collection.find(query)


    def get_all(self):
        return self._collection.find({})


    """{     datos de la mongo
        "video_id": yt.video_id,
        "title": yt.title,
        "keywords": yt.keywords,
        "views": yt.views,
        "length": yt.length,
        "author": yt.author,
        "video_path": file_path,
        "true_label": "Real" / "IA"
        "label": "Real" / "IA",
        "ia_confidence": % o 0 si label es "Real"
        "agreement": boolean    # si los 3 modelos estan de acuerdo en que contiene IA en un punto concreto
    }
    """