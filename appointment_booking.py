"""Simple appointment-booking service used for the coding assessment.

The two external-facing functions are deliberately small stand-ins for real
services.  They can be replaced or monkeypatched by the application/tests.
"""

from __future__ import annotations

from itertools import count
from threading import RLock
from typing import Any


_booked_appointments: set[tuple[str, str, str, str]] = set()
_booked_slots: set[tuple[str, str, str]] = set()
_appointment_numbers = count(1000)
_booking_lock = RLock()


def check_availability(doctor_id: str, date: str, time: str) -> bool:
    """Mock of the availability service."""
    return True


def scheduling_api(
    patient_id: str, doctor_id: str, date: str, time: str
) -> dict[str, str]:
    """Mock of the remote scheduling API."""
    return {"appointment_id": f"APT-{next(_appointment_numbers)}"}


def book_appointment(
    patient_id: str,
    doctor_id: str,
    date: str,
    time: str,
) -> dict[str, Any]:
    """Book a doctor slot and return a consistent JSON-like response.

    A booking is recorded only after the scheduling API reports success, so a
    temporary API problem does not incorrectly make a slot look booked.
    """
    fields = {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "date": date,
        "time": time,
    }
    invalid_field = next(
        (
            name
            for name, value in fields.items()
            if not isinstance(value, str) or not value.strip()
        ),
        None,
    )
    if invalid_field:
        return {"success": False, "error": f"{invalid_field} is required"}

    # Normalise surrounding whitespace before using values as booking keys.
    patient_id, doctor_id, date, time = (
        patient_id.strip(),
        doctor_id.strip(),
        date.strip(),
        time.strip(),
    )
    booking_key = (patient_id, doctor_id, date, time)

    slot_key = (doctor_id, date, time)
    # Keep the duplicate check, availability check, remote scheduling call, and
    # local slot reservation atomic within this process. This prevents two
    # simultaneous requests from reserving the same doctor's slot.
    with _booking_lock:
        if booking_key in _booked_appointments:
            return {"success": False, "error": "Duplicate booking"}
        if slot_key in _booked_slots:
            return {"success": False, "error": "Slot unavailable"}

        try:
            if not check_availability(doctor_id, date, time):
                return {"success": False, "error": "Slot unavailable"}
        except Exception:
            return {"success": False, "error": "Availability check failed"}

        try:
            api_response = scheduling_api(patient_id, doctor_id, date, time)
        except TimeoutError:
            return {"success": False, "error": "Scheduling API timeout"}
        except Exception:
            return {"success": False, "error": "Scheduling API failure"}

        if not isinstance(api_response, dict) or not api_response.get("appointment_id"):
            return {"success": False, "error": "Scheduling API failure"}

        _booked_appointments.add(booking_key)
        _booked_slots.add(slot_key)
        return {
            "success": True,
            "appointment_id": api_response["appointment_id"],
            "message": "Appointment booked successfully",
        }
