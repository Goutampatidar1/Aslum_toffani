class Camera:
    def __init__(
        self,
        camera_name,
        camera_place,
        url,
        company_name,
        company_email,
        company_id,
    ):
        self.company_id = company_id
        self.camera_name = camera_name
        self.camera_place = camera_place
        self.company_name = company_name
        self.company_email = company_email
        self.url = url

    def to_dict(self):
        return {
            "company_id": self.company_id,
            "camera_name": self.camera_name,
            "camera_place": self.camera_place,
            "company_name": self.company_name,
            "company_email": self.company_email,
            "url": self.url,
        }
