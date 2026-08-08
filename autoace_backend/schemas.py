from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AudioAnalysis(BaseModel):
    emotional_tone: Literal[
        "neutral", "satisfied", "frustrated", "upset", "distressed"
    ]
    emotional_intensity: Literal["low", "medium", "high"]
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: Literal["none", "low", "medium", "high"]
    audio_quality: Literal["clear", "slightly_impaired", "severely_impaired"]
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in_seconds: int


class UsageMetadata(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    thinking_tokens: int | None = None


class AnalysisResult(BaseModel):
    name: str
    result: AudioAnalysis
    model: str
    request_seconds: float
    usage: UsageMetadata
