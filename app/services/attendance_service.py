from datetime import datetime
from zoneinfo import ZoneInfo  
from bson import ObjectId
import logging
from app.config import db

IST = ZoneInfo("Asia/Kolkata")


def mark_attendance(user_id, action):
    try:
        try:
            user_oid = ObjectId(user_id)
        except Exception:
            return None, "Invalid user ID format"

        user = db.users.find_one({"_id": user_oid})
        if not user:
            return None, "User not found"

        now = datetime.now(IST)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if action == "checkin":
            check_in_str = now.strftime("%Y-%m-%d %H:%M:%S")
            attendance_doc = {
                "user": user_oid,
                "check_in": check_in_str,
                "check_out": None,
                "working_hours": 0,
            }
            result = db.attendance.insert_one(attendance_doc)
            attendance_id = result.inserted_id

            db.users.update_one(
                {"_id": user_oid}, {"$push": {"total_work": attendance_id}}
            )

            already_exists = False
            for entry in user.get("total_attendence", []):
                entry_date_str = entry.get("date_time")
                if entry_date_str:
                    entry_dt = datetime.strptime(
                        entry_date_str, "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=IST)
                    normalized_entry = entry_dt.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    if normalized_entry == today:
                        already_exists = True
                        break

            if not already_exists:
                db.users.update_one(
                    {"_id": user_oid},
                    {
                        "$push": {
                            "total_attendence": {
                                "date_time": check_in_str,
                            }
                        }
                    },
                )

            attendance_doc["_id"] = str(attendance_id)
            attendance_doc["user"] = str(user_id)
            return attendance_doc, None

        elif action == "checkout":
            attendance = db.attendance.find_one(
                {"user": user_oid, "check_out": None}, sort=[("check_in", -1)]
            )

            if not attendance:
                return None, "No active check-in found to check out"

            check_in_str = attendance["check_in"]
            check_in = datetime.strptime(check_in_str, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=IST
            )
            check_out = datetime.now(IST)
            check_out_str = check_out.strftime("%Y-%m-%d %H:%M:%S")

            working_hours = round((check_out - check_in).total_seconds() / 3600, 2)

            db.attendance.update_one(
                {"_id": attendance["_id"]},
                {
                    "$set": {
                        "check_out": check_out_str,
                        "working_hours": working_hours,
                    }
                },
            )

            attendance["check_out"] = check_out_str
            attendance["working_hours"] = working_hours
            attendance["_id"] = str(attendance["_id"])
            attendance["user"] = str(attendance["user"])

            return attendance, None

        else:
            return None, "Invalid action. Use 'checkin' or 'checkout'"

    except Exception as e:
        logging.error(f"Error in mark_attendance: {e}")
        return None, "Internal server error"
