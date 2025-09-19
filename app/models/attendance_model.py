class Attendance:
    def __init__(self, user_id, check_in, check_out=None, total_hours=0):
        self.user_id = user_id  # reference to the User table
        self.check_in = check_in
        self.check_out = check_out
        self.total_hours = total_hours
        # self.created_at = datetime.now()

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "check_in": self.check_in,
            "check_out": self.check_out,
            "total_hours": self.total_hours,
        }
