from datetime import datetime
from bson import ObjectId
import logging
from app.config import db
from app.models.attendance_model import Attendance


def mark_attendance(unique_user_id, action):
    try:

        try:
            user_oid = unique_user_id
        except Exception:
            return None, "Invalid user ID format"

        # Find user
        user = db.users.find_one({"unique_user_id": unique_user_id})
        if not user:
            return None, "User not found"

        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if action == "checkin":
            # 🛑 Prevent duplicate check-ins
            existing_attendance = db.attendance.find_one(
                {"user_id": user["_id"], "check_out": None}
            )
            if existing_attendance:
                return None, "User already checked in today"

            check_in_str = now.strftime("%Y-%m-%d %H:%M:%S")
            attendance_doc = Attendance(
                user_id=user["_id"],
                check_in=check_in_str,
                check_out=None,
                total_hours=0,
            )
            result = db.attendance.insert_one(attendance_doc.to_dict())
            attendance_id = result.inserted_id

            # Link attendance to user
            db.users.update_one(
                {"unique_user_id": user_oid}, {"$push": {"total_work": attendance_id}}
            )

            # Check if already marked in `total_attendence` (UI/stat tracking maybe)
            already_exists = False
            for entry in user.get("total_attendence", []):
                entry_date_str = entry.get("date_time")
                if entry_date_str:
                    entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d %H:%M:%S")
                    normalized_entry = entry_dt.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    if normalized_entry == today:
                        already_exists = True
                        break

            if not already_exists:
                db.users.update_one(
                    {"_id": user["_id"]},
                    {
                        "$push": {
                            "total_attendence": {
                                "date_time": check_in_str,
                            }
                        }
                    },
                )

            attendance_doc._id = str(attendance_id)
            attendance_doc.user_id = str(user["_id"])

            return attendance_doc, None

        elif action == "checkout":
            attendance = db.attendance.find_one(
                {"user_id": user["_id"], "check_out": None}, sort=[("check_in", -1)]
            )

            if not attendance:
                return None, "No active check-in found to check out"

            check_in_str = attendance["check_in"]
            check_in = datetime.strptime(check_in_str, "%Y-%m-%d %H:%M:%S")
            check_out = datetime.now()
            check_out_str = check_out.strftime("%Y-%m-%d %H:%M:%S")

            total_hours = round((check_out - check_in).total_seconds() / 3600, 2)

            db.attendance.update_one(
                {"_id": attendance["_id"]},
                {
                    "$set": {
                        "check_out": check_out_str,
                        "total_hours": total_hours,
                    }
                },
            )

            return attendance, None

        else:
            return None, "Invalid action. Use 'checkin' or 'checkout'"

    except Exception as e:
        logging.error(f"Error in mark_attendance: {e}")
        return None, "Internal server error"
