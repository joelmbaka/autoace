from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .schemas import LoginRequest, LoginResponse
from .hybrid_analyzer import HybridAnalyzer
from .services import (
    build_validation_summary,
    compare_prediction,
    prepare_batch,
)


load_dotenv()

TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", str(12 * 60 * 60)))
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "evaluator")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "autoace-local")
AUTH_SECRET = os.getenv("AUTH_SECRET", "local-dev-secret-change-me")
BATCH_CONCURRENCY = max(1, int(os.getenv("BATCH_CONCURRENCY", "3")))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="AutoAce Call Analysis Trial API",
    version="0.1.0",
    description="Batch emotional-tone and background-noise analysis for production call audio.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

bearer = HTTPBearer(auto_error=False)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _issue_token(username: str) -> str:
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    encoded_payload = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(
        AUTH_SECRET.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{encoded_payload}.{_b64url_encode(signature)}"


def _verify_token(token: str) -> str:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        supplied_signature = _b64url_decode(encoded_signature)
        expected_signature = hmac.new(
            AUTH_SECRET.encode("utf-8"),
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise ValueError("invalid signature")
        payload = json.loads(_b64url_decode(encoded_payload))
        if int(payload["exp"]) < int(time.time()):
            raise ValueError("expired token")
        username = str(payload["sub"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
        ) from exc
    return username


def require_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required.",
        )
    return _verify_token(credentials.credentials)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        "batch_concurrency": BATCH_CONCURRENCY,
        "auth_defaults_in_use": (
            DASHBOARD_PASSWORD == "autoace-local"
            or AUTH_SECRET == "local-dev-secret-change-me"
        ),
    }


@app.post("/api/v1/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    username_matches = secrets.compare_digest(payload.username, DASHBOARD_USERNAME)
    password_matches = secrets.compare_digest(payload.password, DASHBOARD_PASSWORD)
    if not (username_matches and password_matches):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login credentials.",
        )
    return LoginResponse(
        access_token=_issue_token(payload.username),
        expires_in_seconds=TOKEN_TTL_SECONDS,
    )


def _event(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


@app.post("/api/v1/batches/analyze")
async def analyze_batch(
    files: Annotated[list[UploadFile], File(description="ZIP or folder/file batch")],
    _: Annotated[str, Depends(require_auth)],
) -> StreamingResponse:
    uploaded: list[tuple[str, bytes]] = []
    for upload in files:
        if not upload.filename:
            continue
        uploaded.append((upload.filename, await upload.read()))

    try:
        batch = prepare_batch(uploaded)
        analyzer = HybridAnalyzer()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def stream() -> AsyncIterator[bytes]:
        yield _event(
            {
                "type": "batch_started",
                "total": len(batch.items),
                "processable": sum(item.preflight_error is None for item in batch.items),
                "labeled_count": batch.labeled_count,
                "concurrency": BATCH_CONCURRENCY,
                "warnings": batch.warnings,
                "items": [
                    {
                        "name": item.name,
                        "status": "failed" if item.preflight_error else "pending",
                        "error": item.preflight_error,
                    }
                    for item in batch.items
                ],
            }
        )

        completed = 0
        failed = 0
        comparisons: list[dict[str, Any]] = []
        semaphore = asyncio.Semaphore(BATCH_CONCURRENCY)

        for item in batch.items:
            if item.preflight_error:
                failed += 1
                yield _event(
                    {
                        "type": "file_failed",
                        "name": item.name,
                        "error": item.preflight_error,
                    }
                )

        async def run_item(item: Any) -> tuple[Any, Any, str | None]:
            async with semaphore:
                try:
                    result = await analyzer.analyze(
                        item.name,
                        item.data,
                        item.mime_type,
                    )
                    return item, result, None
                except Exception as exc:  # isolate one malformed/API-failed file from the batch
                    return item, None, str(exc)

        tasks = [
            asyncio.create_task(run_item(item))
            for item in batch.items
            if item.preflight_error is None
        ]

        for task in asyncio.as_completed(tasks):
            item, result, error = await task
            if error is not None:
                failed += 1
                yield _event(
                    {
                        "type": "file_failed",
                        "name": item.name,
                        "error": error,
                    }
                )
                continue

            completed += 1
            validation = None
            if item.expected is not None:
                validation = compare_prediction(result.result, item.expected)
                validation["predicted_tone"] = result.result.emotional_tone
                comparisons.append(validation)

            yield _event(
                {
                    "type": "file_completed",
                    **result.model_dump(),
                    "validation": validation,
                }
            )

        yield _event(
            {
                "type": "batch_completed",
                "completed": completed,
                "failed": failed,
                "total": len(batch.items),
                "validation": build_validation_summary(comparisons),
            }
        )

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
