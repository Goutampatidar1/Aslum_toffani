class Attendance:
    def __init__(
        self,
        user_id,
        check_in,
        check_in_image,
        check_out_image=None,
        check_out=None,
        total_hours=0,
    ):
        self.user_id = user_id  # reference to the User table
        self.check_in = check_in
        self.check_out = check_out
        self.total_hours = total_hours
        self.check_in_image = check_in_image
        self.check_out_image = check_out_image
        # self.created_at = datetime.now()

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "check_in": self.check_in,
            "check_out": self.check_out,
            "total_hours": self.total_hours,
            "check_in_image": self.check_in_image,
            "check_out_image": self.check_out_image,
        }
