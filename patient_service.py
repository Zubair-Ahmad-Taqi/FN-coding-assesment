"""Non-blocking client for the external patient service."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


PATIENT_API_BASE_URL = "https://api.example.com"
DEFAULT_TIMEOUT_SECONDS = 5.0


async def get_patient(
    patient_id: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch one patient without blocking the event loop.

    The caller receives a consistent dictionary for success, HTTP failures, and
    transport failures instead of uncaught requests exceptions.
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        return {"success": False, "patient_id": patient_id, "error": "patient_id is required"}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            base_url=PATIENT_API_BASE_URL,
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
        )

    try:
        response = await client.get(f"/patients/{patient_id.strip()}")
        if response.status_code != httpx.codes.OK:
            return {
                "success": False,
                "patient_id": patient_id,
                "error": f"Patient service returned HTTP {response.status_code}",
            }
        try:
            patient = response.json()
        except ValueError:
            return {
                "success": False,
                "patient_id": patient_id,
                "error": "Patient service returned invalid JSON",
            }
        return {"success": True, "patient_id": patient_id, "patient": patient}
    except httpx.TimeoutException:
        return {"success": False, "patient_id": patient_id, "error": "Patient service timeout"}
    except httpx.RequestError:
        return {"success": False, "patient_id": patient_id, "error": "Patient service request failed"}
    finally:
        if owns_client:
            await client.aclose()


async def get_patients(patient_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch multiple patients concurrently using one reusable async client."""
    timeout = httpx.Timeout(DEFAULT_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(base_url=PATIENT_API_BASE_URL, timeout=timeout) as client:
        results = await asyncio.gather(*(get_patient(patient_id, client) for patient_id in patient_ids))
    return {str(patient_id): result for patient_id, result in zip(patient_ids, results)}
