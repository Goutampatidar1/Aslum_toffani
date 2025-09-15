from app.config import db
from app.models.camera_model import Camera
from app.utils.helper import str_object_id


def create_camera(data):
    camera = Camera(
        camera_name=data["camera_name"],
        camera_place=data["camera_place"],
        url=data["url"],
    )

    result = db.cameras.insert_one(camera.to_dict())
    camera_id = str(result.inserted_id)

    return camera_id, None


def delete_camera(camera_id):
    object_id = str_object_id(camera_id)
    if not object:
        return 0, "Invalid camera ID format"

    camera = db.cameras.find_one({"_id": object_id})
    if not camera:
        return 0, "Camera not found"

    result = db.cameras.delete_one({"_id": object_id})
    if result.deleted_count == 0:
        return 0, "Camera Not Found"

    return result.deleted_count, None


def fetch_all_camera():
    all_cameras = list(db.cameras.find({}))
    if not all_cameras:
        return 0 , "Failed to fetch the camera's"
    
    for camera in all_cameras:
        camera["_id"] = str(camera["_id"])
    return all_cameras , None
