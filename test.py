import datetime
from zoneinfo import ZoneInfo

# Use a specific, timezone-aware datetime for consistent examples
now = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))
print(f"Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")