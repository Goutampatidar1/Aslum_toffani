class User:
    def __init__(
        self,
        name,
        email_id,
        contact_number,
        image,
        total_attendance=None,
        total_work=None,
    ):
        self.name = name
        self.email_id = email_id
        self.contact_number = contact_number
        self.image = image
        self.total_attendance = total_attendance or []  # list of date time
        self.total_work = total_work or []  # list of reference to the atrtendace

    def to_dict(self):
        return {
            "name": self.name,
            "emailId": self.email_id,
            "phoneNumber": self.phone_number,
            "image": self.image,
            "total_attendence": self.total_attendance,
            "total_work": self.total_work,
        }
