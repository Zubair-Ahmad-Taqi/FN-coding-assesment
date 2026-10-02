"""FastAPI application exposing appointment booking."""

from datetime import date as Date
from datetime import time as Time
from typing import Annotated, Literal, Union

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import handle_user_request
from appointment_booking import book_appointment
from rag import answer_with_context


app = FastAPI(title="Appointment API", version="1.0.0")


class AppointmentRequest(BaseModel):
    patient_id: Annotated[str, Field(min_length=1, examples=["PAT-123"])]
    doctor_id: Annotated[str, Field(min_length=1, examples=["DOC-456"])]
    date: Annotated[Date, Field(examples=["2026-10-10"])]
    time: Annotated[Time, Field(examples=["09:30:00"])]


class BookingSuccess(BaseModel):
    success: Literal[True]
    appointment_id: str
    message: str


class BookingFailure(BaseModel):
    success: Literal[False]
    error: str


BookingResponse = Union[BookingSuccess, BookingFailure]


class UserMessageRequest(BaseModel):
    user_message: Annotated[str, Field(min_length=1, examples=[
        "Book patient PAT-123 with Dr. Ahmed on Wednesday at 3 PM."
    ])]


class KnowledgeQueryRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, examples=["When can I cancel an appointment?"])]


def _failure_status(error: str) -> int:
    """Translate domain failures to useful HTTP status codes."""
    if error in {"Slot unavailable", "Duplicate booking"}:
        return status.HTTP_409_CONFLICT
    if error == "Scheduling API timeout":
        return status.HTTP_504_GATEWAY_TIMEOUT
    if error in {"Scheduling API failure", "Availability check failed"}:
        return status.HTTP_502_BAD_GATEWAY
    return status.HTTP_400_BAD_REQUEST


@app.post(
    "/appointments",
    response_model=BookingResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": BookingFailure, "description": "Invalid booking data"},
        409: {"model": BookingFailure, "description": "Slot conflict"},
        502: {"model": BookingFailure, "description": "Upstream service failure"},
        504: {"model": BookingFailure, "description": "Scheduling service timed out"},
    },
    summary="Book an appointment",
)
def create_appointment(payload: AppointmentRequest) -> BookingResponse | JSONResponse:
    """Create an appointment from ISO-8601 date and time values."""
    result = book_appointment(
        patient_id=payload.patient_id,
        doctor_id=payload.doctor_id,
        date=payload.date.isoformat(),
        time=payload.time.isoformat(),
    )

    if result["success"]:
        return result

    return JSONResponse(status_code=_failure_status(result["error"]), content=result)


@app.post("/agent/requests", summary="Process an appointment request through the tool controller")
def process_agent_request(payload: UserMessageRequest) -> dict:
    """Return clarification when details are missing; otherwise invoke the tool."""
    return handle_user_request(payload.user_message)


@app.post("/knowledge/answer", summary="Answer a clinic question using retrieved sources")
def answer_knowledge_question(payload: KnowledgeQueryRequest) -> dict:
    """Return a grounded answer and the chunks used to produce it."""
    return answer_with_context(payload.query)
