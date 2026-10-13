"""Client-safe errors shared by HTTP and WebSocket endpoints."""
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError


class InputError(HTTPException):
    def __init__(self, message: str, *, code: str = 'invalid_request',
                 param: str | None = None, status_code: int = 400,
                 retry_after: int | None = None):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.param = param
        self.retry_after: int | None = retry_after


def error_details(exc):
    if isinstance(exc, (ValidationError, RequestValidationError)):
        first = exc.errors()[0]
        param = '.'.join(str(part) for part in first['loc'] if part not in ('body', 'query'))
        message = first['msg'].removeprefix('Value error, ')
        return {'code': 'invalid_json' if first['type'] == 'json_invalid' else 'invalid_request',
                'param': param, 'message': message, 'retryable': False}
    if isinstance(exc, HTTPException) and exc.status_code < 500:
        details = {'code': getattr(exc, 'code', 'invalid_request'),
                   'param': getattr(exc, 'param', None), 'message': str(exc.detail),
                   'retryable': exc.status_code == 429}
        if isinstance(exc, InputError) and exc.retry_after is not None:
            details['retry_after'] = exc.retry_after
        return details
    return {'code': 'synthesis_failed', 'param': None,
            'message': 'Speech generation failed. Check the server logs.', 'retryable': True}
