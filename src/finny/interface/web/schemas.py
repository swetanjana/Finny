"""Pydantic request/response schemas for Finny web API."""

from typing import Literal, Optional
from pydantic import BaseModel, field_validator


class AskRequest(BaseModel):
    role: str
    query: str

    @field_validator("role")
    @classmethod
    def role_nonempty(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("role must not be empty")
        return v

    @field_validator("query")
    @classmethod
    def query_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v


class FeedbackRequest(BaseModel):
    query_id: int
    rating: Optional[Literal["up", "down"]] = None
    correction_text: Optional[str] = None

    @field_validator("correction_text")
    @classmethod
    def correction_strip(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            return v if v else None
        return v
