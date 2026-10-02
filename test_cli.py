import unittest
from unittest.mock import patch

import agent


class CliRoutingTests(unittest.TestCase):
    def test_routes_booking_message_to_booking_controller(self) -> None:
        with patch("builtins.input", side_effect=["Book patient PAT-1 with Dr. Ahmed on Friday at 10 AM.", "quit"]), patch.object(
            agent, "BookingConversation"
        ) as conversation_class, patch.object(agent, "answer_with_context") as rag:
            conversation_class.return_value.handle.side_effect = [
                {
                    "success": True,
                    "action": "book_appointment",
                    "result": {
                        "success": True,
                        "appointment_id": "APT-1",
                        "message": "Appointment booked successfully",
                    },
                }
            ]
            agent.run_cli()

        conversation_class.return_value.handle.assert_called_once()
        rag.assert_not_called()
