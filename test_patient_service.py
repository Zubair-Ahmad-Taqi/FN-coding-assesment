import asyncio
import unittest

import httpx

from patient_service import get_patient


class PatientServiceTests(unittest.TestCase):
    def test_returns_clean_failure_for_non_200_response(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(404, json={"detail": "not found"})
        )

        async def run() -> dict:
            async with httpx.AsyncClient(transport=transport, base_url="https://test") as client:
                return await get_patient("PAT-404", client)

        result = asyncio.run(run())
        self.assertEqual(result["error"], "Patient service returned HTTP 404")

    def test_returns_clean_failure_for_request_error(self) -> None:
        def fail(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline")

        async def run() -> dict:
            async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
                return await get_patient("PAT-1", client)

        result = asyncio.run(run())
        self.assertEqual(result["error"], "Patient service request failed")
