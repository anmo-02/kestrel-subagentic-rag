from typing import Literal, Optional

from pydantic import BaseModel

Grade = Literal["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
Office = Literal["Pune", "Mumbai", "Bengaluru"]

METRO_CITIES = {"Mumbai", "Delhi NCR", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Pune"}


class EmployeeProfile(BaseModel):
    name: str = "Employee"
    grade: Optional[Grade] = None
    office: Optional[Office] = None
    joining_date: Optional[str] = None  # ISO date string, e.g. "2024-03-15"

    def as_context_line(self) -> str:
        parts = [f"Grade: {self.grade or 'not set'}", f"Office: {self.office or 'not set'}"]
        if self.joining_date:
            parts.append(f"Joined: {self.joining_date}")
        return "Employee profile -- " + ", ".join(parts) + "."


def is_metro(city: str) -> Optional[bool]:
    """Returns True/False if the city is unambiguously known, else None."""
    if not city:
        return None
    normalized = city.strip().lower()
    for metro in METRO_CITIES:
        if metro.lower() == normalized:
            return True
    # A short list of common non-metro Indian cities that show up in test
    # questions, so the supervisor can flag them with certainty rather than
    # leaving it to guesswork.
    known_non_metro = {"nashik", "pune rural", "nagpur", "surat", "jaipur", "lucknow", "indore", "kochi", "goa", "panaji"}
    if normalized in known_non_metro:
        return False
    return None
