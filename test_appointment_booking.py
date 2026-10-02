import unittest
from unittest.mock import patch

import appointment_booking as booking


class BookAppointmentTests(unittest.TestCase):
    def setUp(self) -> None:
        """Keep the module's in-memory duplicate tracker isolated per test."""
        booking._booked_appointments.clear()
        booking._booked_slots.clear()

    def test_returns_failure_when_slot_is_unavailable(self) -> None:
        with patch.object(booking, "check_availability", return_value=False):
            result = booking.book_appointment("P-1", "D-1", "2026-10-10", "09:00")

        self.assertEqual(result, {"success": False, "error": "Slot unavailable"})

    def test_returns_failure_when_scheduling_api_times_out(self) -> None:
        with patch.object(booking, "scheduling_api", side_effect=TimeoutError):
            result = booking.book_appointment("P-1", "D-1", "2026-10-10", "09:00")

        self.assertEqual(result, {"success": False, "error": "Scheduling API timeout"})

    def test_prevents_a_duplicate_booking(self) -> None:
        with patch.object(
            booking, "scheduling_api", return_value={"appointment_id": "APT-1007"}
        ):
            booking.book_appointment("P-1", "D-1", "2026-10-10", "09:00")
            result = booking.book_appointment("P-1", "D-1", "2026-10-10", "09:00")

        self.assertEqual(result, {"success": False, "error": "Duplicate booking"})

    def test_prevents_two_patients_from_booking_the_same_doctor_slot(self) -> None:
        with patch.object(
            booking, "scheduling_api", return_value={"appointment_id": "APT-1007"}
        ):
            booking.book_appointment("P-1", "D-1", "2026-10-10", "09:00")
            result = booking.book_appointment("P-2", "D-1", "2026-10-10", "09:00")

        self.assertEqual(result, {"success": False, "error": "Slot unavailable"})
