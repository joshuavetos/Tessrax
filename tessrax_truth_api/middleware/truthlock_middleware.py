"""Truth-Lock middleware enforcing provenance requirements."""
from __future__ import annotations

from typing import Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from tessrax_truth_api.utils import decode_jwt


class TruthLockMiddleware(BaseHTTPMiddleware):
    """Ensure requests include a valid JWT token."""

    def __init__(self, app, *, protected_paths: tuple[str, ...] = ("/detect",)):
        super().__init__(app)
        self._protected = protected_paths

    async def dispatch(self, request: Request, call_next: Callable):
        if any(request.url.path.startswith(path) for path in self._protected):
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.lower().startswith("bearer "):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
            token = auth_header.split(" ", 1)[1]
            claims = decode_jwt(token)
            request.state.jwt_claims = claims
            request.state.bearer_token = token
        return await call_next(request)
