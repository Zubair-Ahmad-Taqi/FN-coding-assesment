"""A small, deterministic controller that demonstrates an LLM tool flow.

In production, an LLM would select ``BOOK_APPOINTMENT_TOOL`` and return its
arguments.  This assessment version uses lightweight extraction so it runs
without an API key while retaining the same safety boundary: the booking tool
is called only after all fields are present.
"""

from __future__ import annotations

import re
import logging
import time
import argparse
import json
from typing import Any
from uuid import uuid4

from appointment_booking import book_appointment
from ollama_client import OllamaError, chat as ollama_chat
from rag import answer_with_context


logger = logging.getLogger("clinic.agent")


BOOK_APPOINTMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "book_appointment",
        "description": "Book a patient with a doctor at a specific date and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient identifier"},
                "doctor_id": {"type": "string", "description": "Doctor identifier"},
                "date": {"type": "string", "description": "Requested appointment date"},
                "time": {"type": "string", "description": "Requested appointment time"},
            },
            "required": ["patient_id", "doctor_id", "date", "time"],
            "additionalProperties": False,
        },
    },
}

_DOCTOR_IDS = {"ahmed": "DOC-AHMED"}
_WEEKDAYS = "monday|wednesday|friday"


def is_booking_request(message: str) -> bool:
    """Return whether a message appears to be an appointment action."""
    return bool(re.search(r"\b(book|schedule|appointment)\b", message, re.I))


def _extract_fields(message: str) -> dict[str, str]:
    """Extract only explicit values; no appointment details are guessed."""
    fields: dict[str, str] = {}
    patient = re.search(r"\b(?:patient(?:\s+id)?|id)\s*(?:is|:|#)?\s*([\w-]+)", message, re.I)
    if patient:
        fields["patient_id"] = patient.group(1)

    doctor = re.search(r"\bdr\.?\s+([a-z]+)\b", message, re.I)
    if doctor:
        doctor_name = doctor.group(1).lower()
        fields["doctor_id"] = _DOCTOR_IDS.get(doctor_name, f"DOC-{doctor_name.upper()}")

    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", message)
    weekday_match = re.search(rf"\b({_WEEKDAYS})\b", message, re.I)
    if date_match:
        fields["date"] = date_match.group(0)
    elif weekday_match:
        fields["date"] = weekday_match.group(1).title()

    time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b", message, re.I)
    if time_match:
        hour, minute, meridiem = time_match.groups()
        hour_number = int(hour) % 12 + (12 if meridiem.lower() == "p" else 0)
        fields["time"] = f"{hour_number:02d}:{minute or '00'}"
    else:
        twenty_four_hour = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", message)
        if twenty_four_hour:
            fields["time"] = twenty_four_hour.group(0)
    return fields


def _tool_arguments_from_ollama(
    message: str, known_fields: dict[str, str] | None = None
) -> dict[str, str] | None:
    """Ask the local model to select the booking tool and provide its arguments."""
    try:
        response = ollama_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Use book_appointment only if every required value is explicit in "
                        "the user message or confirmed conversation details. Never invent a "
                        "patient ID, doctor, date, or time. Confirmed details: "
                        f"{known_fields or {}}"
                    ),
                },
                {"role": "user", "content": message},
            ],
            tools=[BOOK_APPOINTMENT_TOOL],
        )
    except OllamaError:
        return None

    tool_calls = response["message"].get("tool_calls", [])
    if not tool_calls:
        return None
    function = tool_calls[0].get("function", {})
    arguments = function.get("arguments")
    if function.get("name") != "book_appointment" or not isinstance(arguments, dict):
        return None
    return {key: value for key, value in arguments.items() if isinstance(value, str)}


def handle_user_request(
    user_message: str, known_fields: dict[str, str] | None = None
) -> dict[str, Any]:
    """Route a booking request safely and return a structured final response."""
    if not isinstance(user_message, str) or not user_message.strip():
        return {
            "success": False,
            "action": "clarification_needed",
            "missing_fields": ["user_message"],
            "message": "Please provide your appointment request.",
        }

    if not is_booking_request(user_message) and not known_fields:
        return {
            "success": False,
            "action": "unsupported_request",
            "message": "I can help book an appointment. Please provide the patient ID, doctor, date, and time.",
        }

    # Deterministic extraction first lets us ask for missing data immediately.
    # When complete, Ollama is asked to issue the actual tool call.
    fields = dict(known_fields or {})
    fields.update(_extract_fields(user_message))
    required = ("patient_id", "doctor_id", "date", "time")
    missing = [field for field in required if field not in fields]
    if missing:
        return {
            "success": False,
            "action": "clarification_needed",
            "missing_fields": missing,
            "message": "Please provide: " + ", ".join(missing) + ". Use YYYY-MM-DD for a calendar date.",
        }

    tool_arguments = _tool_arguments_from_ollama(user_message, known_fields=fields)
    if tool_arguments is not None:
        model_missing = [field for field in required if not tool_arguments.get(field, "").strip()]
        if model_missing:
            return {
                "success": False,
                "action": "clarification_needed",
                "missing_fields": model_missing,
                "message": "Please provide: " + ", ".join(model_missing) + ".",
            }
        # Fields collected from the user across turns remain authoritative; the
        # model's tool call is a request to invoke this tool, not permission to
        # replace confirmed values with guessed ones.
        fields = {field: fields[field].strip() for field in required}

    # This is the only path that invokes the backend booking tool, and only
    # after the required tool arguments are present. The trace log provides a
    # minimal audit trail for tool selection, inputs, outcome, and latency.
    trace_id = str(uuid4())
    started_at = time.perf_counter()
    logger.info(
        "trace_id=%s tool=book_appointment tool_arguments=%s",
        trace_id,
        fields,
    )
    try:
        booking_result = book_appointment(**fields)
    except Exception as error:  # Last-resort controller boundary.
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.exception(
            "trace_id=%s tool=book_appointment error=%s latency_ms=%s",
            trace_id,
            error,
            latency_ms,
        )
        return {
            "success": False,
            "action": "book_appointment",
            "trace_id": trace_id,
            "result": {"success": False, "error": "Booking tool failure"},
        }

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "trace_id=%s tool=book_appointment tool_result=%s latency_ms=%s",
        trace_id,
        booking_result,
        latency_ms,
    )
    return {
        "success": booking_result["success"],
        "action": "book_appointment",
        "trace_id": trace_id,
        "result": booking_result,
    }


class BookingConversation:
    """Stores explicitly supplied booking details for one terminal chat session."""

    def __init__(self) -> None:
        self.pending_fields: dict[str, str] = {}
        self.active = False

    def handle(self, user_message: str) -> dict[str, Any] | None:
        """Handle a booking turn, or return ``None`` for a knowledge question."""
        if not self.active and not is_booking_request(user_message):
            return None

        self.active = True
        # Only merge explicit values from the current message. Previously
        # collected fields stay available until a booking reaches a final state.
        self.pending_fields.update(_extract_fields(user_message))
        response = handle_user_request(user_message, known_fields=self.pending_fields)

        if response.get("action") == "book_appointment":
            self.pending_fields.clear()
            self.active = False
        return response


def _display_response(response: dict[str, Any], show_tool_response: bool) -> None:
    """Show a conversational reply; raw tool data is opt-in for debugging."""
    if response.get("action") == "book_appointment":
        tool_result = response["result"]
        if tool_result["success"]:
            print(f"Assistant: {tool_result['message']} (ID: {tool_result['appointment_id']})")
        else:
            print(f"Assistant: I couldn't book that appointment: {tool_result['error']}.")
    else:
        print(f"Assistant: {response['message'] if 'message' in response else response['answer']}")

    if show_tool_response:
        print("\nTool response:")
        print(json.dumps(response, indent=2))


def run_cli(show_tool_response: bool = False) -> None:
    """Run the interactive terminal chatbot until the user exits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    print("Clinic assistant is ready. Ask a clinic question or book an appointment.")
    print("Include patient ID, doctor, date, and time when booking. Type 'quit' to exit.")
    booking_conversation = BookingConversation()

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if user_message.lower() in {"quit", "exit"}:
            print("Goodbye.")
            return
        if not user_message:
            print("Assistant: Please enter a question or appointment request.")
            continue

        response = booking_conversation.handle(user_message)
        if response is None:
            response = answer_with_context(user_message)
        _display_response(response, show_tool_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the clinic assistant.")
    parser.add_argument(
        "--show-tool-response",
        action="store_true",
        help="Also print the raw structured agent/tool response after each reply.",
    )
    arguments = parser.parse_args()
    run_cli(show_tool_response=arguments.show_tool_response)
