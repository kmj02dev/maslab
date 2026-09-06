"""Shared HTTP model backend constants."""


RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
