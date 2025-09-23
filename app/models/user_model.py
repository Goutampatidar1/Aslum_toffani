import uuid


class User:
    def __init__(
        self,
        name,
        email_id,
        contact_number,
        images,
        major,
        unique_user_id,
        company_id,
        company_user_id,
        total_attendance=None,
        total_work=None,
    ):
        self.name = name
        self.email_id = email_id
        self.contact_number = contact_number
        self.images = images
        self.major = major
        self.total_attendance = (
            total_attendance or []
        )  
        self.total_work = total_work or []  
        self.unique_user_id = unique_user_id
        self.company_id = company_id
        self.company_user_id = company_user_id

    def to_dict(self):
        return {
            "name": self.name,
            "email_id": self.email_id,
            "contact_number": self.contact_number,
            "images": self.images,
            "major": self.major,
            "total_attendence": self.total_attendance,
            "total_work": self.total_work,
            "unique_user_id": self.unique_user_id,
            "company_id" : self.company_id,
            "company_user_id" : self.company_user_id
        }
