from bson.objectid import (ObjectId  , InvalidId)

def str_object_id(id_str):
    try:
        return ObjectId(id_str)
    except InvalidId:
        return None