class User:
    def __init__(
        self,
        name,
        email_id,
        contact_number,
        image,
        major,
        total_attendance=None,
        total_work=None,
    ):
        self.name = name
        self.email_id = email_id
        self.contact_number = contact_number
        self.image = image
        self.major = major
        self.total_attendance = (
            total_attendance or []
        )  # list of date time and total work
        self.total_work = total_work or []  # list of reference to the atrtendace

    def to_dict(self):
        return {
            "name": self.name,
            "email_id": self.email_id,
            "contact_number": self.contact_number,
            "image": self.image,
            "major": self.major,
            "total_attendence": self.total_attendance,
            "total_work": self.total_work,
        }
