from app.config import db
import logging


def get_user_details_by_unique_id(unique_user_id):
    if not unique_user_id:
        logging.error("Unique user id not passed to this function")
        return None

    user_details = db.users.find_one({"unique_user_id": str(unique_user_id)})

    print("USER DETAILS" , user_details)
    if user_details:
        logging.info("User Details found returning to detection app", user_details)
        return user_details

    return None


