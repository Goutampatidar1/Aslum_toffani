class Camera:
    def __init__(self, camera_name, camera_place, url, comapny_id=None):
        self.company_id = comapny_id
        self.camera_name = camera_name
        self.camera_place = camera_place
        self.url = url

    def to_dict(self):
        return {
            "company_id": self.company_id,
            "camera_name": self.camera_name,
            "camera_place": self.camera_place,
            "url": self.url,
        }
