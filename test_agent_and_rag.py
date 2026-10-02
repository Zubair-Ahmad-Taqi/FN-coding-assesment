import unittest
from unittest.mock import patch

import agent
from rag import answer_with_context, documents, retrieve


class AgentTests(unittest.TestCase):
    def test_asks_for_missing_patient_id_without_booking(self) -> None:
        with patch.object(agent, "book_appointment") as book:
            result = agent.handle_user_request("Book me with Dr. Ahmed on Wednesday at 3 PM.")

        self.assertEqual(result["action"], "clarification_needed")
        self.assertIn("patient_id", result["missing_fields"])
        book.assert_not_called()

    def test_books_only_after_all_fields_are_extracted(self) -> None:
        confirmed = {"success": True, "appointment_id": "APT-1007", "message": "Appointment booked successfully"}
        tool_arguments = {"patient_id": "PAT-1", "doctor_id": "DOC-AHMED", "date": "Wednesday", "time": "15:00"}
        with patch.object(agent, "book_appointment", return_value=confirmed) as book, patch.object(
            agent, "_tool_arguments_from_ollama", return_value=tool_arguments
        ):
            result = agent.handle_user_request(
                "Book patient PAT-1 with Dr. Ahmed on Wednesday at 3 PM."
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["result"], confirmed)
        book.assert_called_once_with(
            patient_id="PAT-1", doctor_id="DOC-AHMED", date="Wednesday", time="15:00"
        )

    def test_conversation_remembers_booking_details_across_turns(self) -> None:
        conversation = agent.BookingConversation()

        first = conversation.handle("Book my appointment with Dr. Ahmed for wednesdat at 9 AM")
        second = conversation.handle("My patient ID is PAT-123 and the date is 02/10/2026")

        self.assertEqual(first["missing_fields"], ["patient_id", "date"])
        self.assertEqual(second["missing_fields"], ["date"])
        self.assertEqual(conversation.pending_fields["doctor_id"], "DOC-AHMED")
        self.assertEqual(conversation.pending_fields["time"], "09:00")
        self.assertEqual(conversation.pending_fields["patient_id"], "PAT-123")

    def test_logs_tool_arguments_result_and_latency_with_trace_id(self) -> None:
        tool_arguments = {
            "patient_id": "PAT-1",
            "doctor_id": "DOC-AHMED",
            "date": "Wednesday",
            "time": "09:00",
        }
        tool_result = {
            "success": True,
            "appointment_id": "APT-1",
            "message": "Appointment booked successfully",
        }
        with patch.object(agent, "_tool_arguments_from_ollama", return_value=tool_arguments), patch.object(
            agent, "book_appointment", return_value=tool_result
        ), self.assertLogs("clinic.agent", level="INFO") as logs:
            result = agent.handle_user_request(
                "Book patient PAT-1 with Dr. Ahmed on Wednesday at 9 AM."
            )

        self.assertTrue(result["trace_id"])
        log_output = "\n".join(logs.output)
        self.assertIn("tool_arguments", log_output)
        self.assertIn("tool_result", log_output)
        self.assertIn("latency_ms", log_output)


class RetrievalTests(unittest.TestCase):
    def test_answers_doctor_name_question_from_retrieved_context(self) -> None:
        with patch("rag._answer_with_ollama", return_value="The doctor's name is Dr. Ahmed."):
            result = answer_with_context("What is the doctor's name?")

        self.assertTrue(result["success"])
        self.assertIn("Dr. Ahmed", result["answer"])
        self.assertIn(documents[1], result["sources"])

    def test_requests_clarification_for_likely_misspelling(self) -> None:
        result = answer_with_context("What is doctor naem?")

        self.assertFalse(result["success"])
        self.assertIn("Did you mean 'name'", result["answer"])

    def test_retrieves_relevant_cancellation_policy(self) -> None:
        results = retrieve("How long before can I cancel?", documents)
        self.assertIn(documents[0], results)

    def test_does_not_invent_when_no_source_is_relevant(self) -> None:
        result = answer_with_context("What is the clinic parking fee?")
        self.assertFalse(result["success"])
        self.assertEqual(result["sources"], [])
