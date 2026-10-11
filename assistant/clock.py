"""Local time and date, and the time in other places ("what time is it in Tokyo?").

Worked out on this computer from the time zone database (the tzdata
package), not by the LLM, which gets time-zone sums wrong.
"""

import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Spoken place -> IANA zone. Countries with one main zone, and big cities.
PLACES = {
    "india": "Asia/Kolkata", "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata", "bengaluru": "Asia/Kolkata", "chennai": "Asia/Kolkata", "vellore": "Asia/Kolkata",
    "hyderabad": "Asia/Kolkata", "kolkata": "Asia/Kolkata", "pune": "Asia/Kolkata",
    "japan": "Asia/Tokyo", "tokyo": "Asia/Tokyo", "osaka": "Asia/Tokyo",
    "china": "Asia/Shanghai", "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai", "hong kong": "Asia/Hong_Kong",
    "korea": "Asia/Seoul", "south korea": "Asia/Seoul", "seoul": "Asia/Seoul",
    "singapore": "Asia/Singapore", "dubai": "Asia/Dubai", "uae": "Asia/Dubai", "abu dhabi": "Asia/Dubai",
    "pakistan": "Asia/Karachi", "karachi": "Asia/Karachi", "bangladesh": "Asia/Dhaka", "dhaka": "Asia/Dhaka",
    "nepal": "Asia/Kathmandu", "sri lanka": "Asia/Colombo", "thailand": "Asia/Bangkok", "bangkok": "Asia/Bangkok",
    "indonesia": "Asia/Jakarta", "jakarta": "Asia/Jakarta", "philippines": "Asia/Manila", "manila": "Asia/Manila",
    "vietnam": "Asia/Ho_Chi_Minh", "malaysia": "Asia/Kuala_Lumpur", "kuala lumpur": "Asia/Kuala_Lumpur",
    "saudi arabia": "Asia/Riyadh", "riyadh": "Asia/Riyadh", "qatar": "Asia/Qatar", "doha": "Asia/Qatar",
    "israel": "Asia/Jerusalem", "turkey": "Europe/Istanbul", "istanbul": "Europe/Istanbul",
    "uk": "Europe/London", "england": "Europe/London", "britain": "Europe/London", "london": "Europe/London",
    "ireland": "Europe/Dublin", "dublin": "Europe/Dublin", "france": "Europe/Paris", "paris": "Europe/Paris",
    "germany": "Europe/Berlin", "berlin": "Europe/Berlin", "munich": "Europe/Berlin", "spain": "Europe/Madrid",
    "madrid": "Europe/Madrid", "italy": "Europe/Rome", "rome": "Europe/Rome", "netherlands": "Europe/Amsterdam",
    "amsterdam": "Europe/Amsterdam", "switzerland": "Europe/Zurich", "zurich": "Europe/Zurich",
    "sweden": "Europe/Stockholm", "russia": "Europe/Moscow", "moscow": "Europe/Moscow",
    "egypt": "Africa/Cairo", "cairo": "Africa/Cairo", "nigeria": "Africa/Lagos", "lagos": "Africa/Lagos",
    "kenya": "Africa/Nairobi", "nairobi": "Africa/Nairobi", "south africa": "Africa/Johannesburg",
    "new york": "America/New_York", "boston": "America/New_York", "washington": "America/New_York",
    "miami": "America/New_York", "toronto": "America/Toronto", "chicago": "America/Chicago",
    "texas": "America/Chicago", "dallas": "America/Chicago", "houston": "America/Chicago",
    "denver": "America/Denver", "california": "America/Los_Angeles", "los angeles": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles", "silicon valley": "America/Los_Angeles",
    "vancouver": "America/Vancouver", "mexico": "America/Mexico_City", "mexico city": "America/Mexico_City",
    "brazil": "America/Sao_Paulo", "sao paulo": "America/Sao_Paulo", "argentina": "America/Argentina/Buenos_Aires",
    "australia": "Australia/Sydney", "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "perth": "Australia/Perth", "new zealand": "Pacific/Auckland", "auckland": "Pacific/Auckland",
}


def _say_time(moment: datetime) -> str:
    return moment.strftime("%I:%M %p").lstrip("0")


def now_local() -> str:
    return f"It's {_say_time(datetime.now())}."


def today() -> str:
    return f"Today is {datetime.now().strftime('%A, %d %B %Y').replace(' 0', ' ')}."


def time_in(place: str) -> str | None:
    """'It's 12:32 AM on Monday in Tokyo.' None if the place isn't known."""
    key = re.sub(r"[^a-z ]", "", place.lower()).strip()
    key = re.sub(r"^the ", "", key)
    zone = PLACES.get(key)
    if zone is None:
        return None
    try:
        there = datetime.now(ZoneInfo(zone))
    except ZoneInfoNotFoundError:
        return None
    here = datetime.now().astimezone()
    day = "" if there.date() == here.date() else f" on {there.strftime('%A')}"
    return f"It's {_say_time(there)}{day} in {place.title()}."
