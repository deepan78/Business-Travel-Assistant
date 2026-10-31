import math
from langchain_core.tools import tool


# Hotel reimbursement caps in INR per night
HOTEL_CAPS = {
    "A": {
        "metro": 4000.00,
        "other": 3000.00
    },
    "B": {
        "metro": 3000.00,
        "other": 2200.00
    },
    "C": {
        "metro": 2000.00,
        "other": 1500.00
    }
}


@tool
def estimate_hotel_reimbursement(
    grade: str,
    city_category: str,
    nights: int,
    nightly_charge: float
) -> dict:
    """
    Estimate hotel reimbursement using employee grade,
    city category, number of nights, and nightly hotel charge.

    Grade must be A, B, or C.
    City category must be metro or other.
    Nights must be at least 1.
    Nightly charge must be zero or greater.

    Returns the nightly cap, actual total, eligible estimate,
    and excess amount in INR.
    """

    grade = grade.strip().upper()
    city_category = city_category.strip().lower()

    # Validate employee grade
    if grade not in HOTEL_CAPS:
        raise ValueError("Grade must be A, B, or C.")

    # Validate city category
    if city_category not in ("metro", "other"):
        raise ValueError("City category must be metro or other.")

    # Validate number of nights
    if isinstance(nights, bool) or not isinstance(nights, int):
        raise ValueError("Nights must be a whole number.")

    if nights < 1:
        raise ValueError("Nights must be at least 1.")

    # Validate nightly charge
    if isinstance(nightly_charge, bool):
        raise ValueError("Nightly charge must be a number.")

    try:
        nightly_charge = float(nightly_charge)
    except (TypeError, ValueError):
        raise ValueError("Nightly charge must be a valid number.")

    if not math.isfinite(nightly_charge) or nightly_charge < 0:
        raise ValueError("Nightly charge must be zero or greater.")

    # Retrieve the applicable nightly cap
    nightly_cap = HOTEL_CAPS[grade][city_category]

    # Calculate reimbursement estimates
    actual_total = nights * nightly_charge
    eligible_estimate = nights * min(nightly_charge, nightly_cap)
    excess_amount = actual_total - eligible_estimate

    return {
        "nightly_cap": round(nightly_cap, 2),
        "actual_total": round(actual_total, 2),
        "eligible_estimate": round(eligible_estimate, 2),
        "excess_amount": round(excess_amount, 2)
    }
