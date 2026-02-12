"""Chamber platform API client.

Provides authenticated access to the Chamber platform for uploading,
retrieving, comparing, and managing GPU profiling results.  Uses only
``urllib.request`` from the standard library so there are no external HTTP
dependencies.

Retry behaviour:
    Transient server errors (HTTP 5xx) are retried up to three times with
    exponential backoff (1 s, 2 s, 4 s).  Client errors (4xx) are raised
    immediately.

Authentication:
    All requests include an ``Authorization: Bearer <api_key>`` header.
    The API key can be provided directly or read from the
    ``CHAMBER_API_KEY`` environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------
_ENV_API_KEY = "CHAMBER_API_KEY"
_ENV_API_URL = "CHAMBER_API_URL"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_API_URL = "https://api.usechamber.io"
_DEFAULT_TIMEOUT_S = 30
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_BACKOFF_MULTIPLIER = 2.0


# ============================================================================
# Exception
# ============================================================================


class ChamberAPIError(Exception):
    """Raised when the Chamber API returns an error response.

    Attributes
    ----------
    status_code:
        HTTP status code returned by the API, or ``0`` if the error
        occurred before a response was received (e.g. network failure).
    message:
        Human-readable error description.
    """

    def __init__(self, message: str, status_code: int = 0) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{status_code}] {message}" if status_code else message)


# ============================================================================
# Client
# ============================================================================


class ChamberClient:
    """Authenticated client for the Chamber platform REST API.

    Usage::

        client = ChamberClient(api_key="ck_live_...")
        url = client.upload_profile(profile_data)
        print("View your profile at:", url)

        profiles = client.list_profiles(job_id="job-abc", limit=10)
        comparison = client.compare_profiles(profiles[0]["id"], profiles[1]["id"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        self._api_key: str = (
            api_key
            or os.environ.get(_ENV_API_KEY, "")
        )
        if not self._api_key:
            raise ChamberAPIError(
                "Chamber API key is required. Provide the api_key argument "
                f"or set the {_ENV_API_KEY} environment variable."
            )

        self._api_url: str = (
            api_url
            or os.environ.get(_ENV_API_URL, "")
            or _DEFAULT_API_URL
        ).rstrip("/")

        logger.debug(
            "ChamberClient initialised (api_url=%s).", self._api_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_profile(
        self,
        profile_data: dict,
        job_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Upload a profile to the Chamber platform.

        Parameters
        ----------
        profile_data:
            The full profile JSON payload (as produced by the profiler
            suite or the ``ReportGenerator``).
        job_id:
            Optional Chamber job ID to associate the profile with.
        tags:
            Optional list of string tags for filtering and organisation.

        Returns
        -------
        str
            The URL where the uploaded profile can be viewed.

        Raises
        ------
        ChamberAPIError
            If the upload fails.
        """
        body: Dict[str, Any] = {"profile_data": profile_data}
        if job_id is not None:
            body["job_id"] = job_id
        if tags is not None:
            body["tags"] = tags

        response = self._request("POST", "/v1/profiles", json_body=body)
        profile_url: str = response.get("url", response.get("profile_url", ""))
        if not profile_url:
            # Construct a fallback URL from the profile ID if the server
            # did not include one in the response.
            profile_id = response.get("id", response.get("profile_id", ""))
            if profile_id:
                profile_url = f"{self._api_url}/v1/profiles/{profile_id}"

        logger.info("Profile uploaded successfully: %s", profile_url)
        return profile_url

    def get_profile(self, profile_id: str) -> dict:
        """Retrieve a single profile by its ID.

        Parameters
        ----------
        profile_id:
            The unique identifier of the profile.

        Returns
        -------
        dict
            The full profile data as returned by the API.

        Raises
        ------
        ChamberAPIError
            If the profile is not found or another error occurs.
        """
        return self._request("GET", f"/v1/profiles/{profile_id}")

    def list_profiles(
        self,
        job_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """List profiles, optionally filtered by job.

        Parameters
        ----------
        job_id:
            If provided, only profiles associated with this job are
            returned.
        limit:
            Maximum number of profiles to return (default 20).

        Returns
        -------
        list[dict]
            A list of profile summary dictionaries.

        Raises
        ------
        ChamberAPIError
            On API errors.
        """
        params: Dict[str, str] = {"limit": str(limit)}
        if job_id is not None:
            params["job_id"] = job_id

        response = self._request("GET", "/v1/profiles", params=params)

        # The API may wrap the list in a {"profiles": [...]} envelope.
        if isinstance(response, dict) and "profiles" in response:
            profiles: List[dict] = response["profiles"]
            return profiles
        if isinstance(response, list):
            return response  # type: ignore[return-value]
        return [response]

    def compare_profiles(
        self,
        profile_id_1: str,
        profile_id_2: str,
    ) -> dict:
        """Request a server-side comparison of two profiles.

        Parameters
        ----------
        profile_id_1:
            ID of the first (baseline) profile.
        profile_id_2:
            ID of the second (comparison) profile.

        Returns
        -------
        dict
            Comparison results including deltas for key metrics.

        Raises
        ------
        ChamberAPIError
            On API errors.
        """
        body = {
            "profile_id_1": profile_id_1,
            "profile_id_2": profile_id_2,
        }
        return self._request("POST", "/v1/profiles/compare", json_body=body)

    def attach_to_job(self, profile_id: str, job_id: str) -> None:
        """Associate an existing profile with a Chamber job.

        Parameters
        ----------
        profile_id:
            The profile to attach.
        job_id:
            The job to attach it to.

        Raises
        ------
        ChamberAPIError
            On API errors.
        """
        body = {"job_id": job_id}
        self._request("POST", f"/v1/profiles/{profile_id}/attach", json_body=body)
        logger.info(
            "Profile %s attached to job %s.", profile_id, job_id,
        )

    def get_historical_profiles(
        self,
        job_id: Optional[str] = None,
        days: int = 30,
    ) -> List[dict]:
        """Retrieve historical profiles for trend analysis.

        Parameters
        ----------
        job_id:
            If provided, only profiles associated with this job are
            returned.
        days:
            Number of days of history to retrieve (default 30).

        Returns
        -------
        list[dict]
            A list of profile summary dictionaries ordered by timestamp.

        Raises
        ------
        ChamberAPIError
            On API errors.
        """
        params: Dict[str, str] = {"days": str(days)}
        if job_id is not None:
            params["job_id"] = job_id

        response = self._request("GET", "/v1/profiles/history", params=params)

        if isinstance(response, dict) and "profiles" in response:
            profiles: List[dict] = response["profiles"]
            return profiles
        if isinstance(response, list):
            return response  # type: ignore[return-value]
        return [response]

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Execute an HTTP request against the Chamber API.

        Handles authentication, JSON serialisation, retries with
        exponential backoff for transient (5xx) errors, and response
        parsing.

        Parameters
        ----------
        method:
            HTTP method (``"GET"``, ``"POST"``, etc.).
        path:
            API path (e.g. ``"/v1/profiles"``).
        json_body:
            Optional request body, serialised as JSON.
        params:
            Optional query-string parameters.

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        ChamberAPIError
            On HTTP errors after all retries are exhausted, or on
            non-retryable (4xx) errors.
        """
        url = self._api_url + path
        if params:
            url = url + "?" + urllib.parse.urlencode(params)

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": "chamber-profiler/1.0",
        }

        data: Optional[bytes] = None
        if json_body is not None:
            data = json.dumps(json_body, default=str).encode("utf-8")
            headers["Content-Type"] = "application/json"

        last_error: Optional[ChamberAPIError] = None
        backoff = _INITIAL_BACKOFF_S

        for attempt in range(1, _MAX_RETRIES + 1):
            request = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method=method.upper(),
            )

            try:
                logger.debug(
                    "Chamber API request: %s %s (attempt %d/%d)",
                    method.upper(), url, attempt, _MAX_RETRIES,
                )
                with urllib.request.urlopen(
                    request, timeout=_DEFAULT_TIMEOUT_S,
                ) as response:
                    return self._handle_response(response)

            except urllib.error.HTTPError as exc:
                status_code = exc.code
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass

                error_message = self._parse_error_message(
                    error_body, status_code,
                )
                last_error = ChamberAPIError(
                    message=error_message, status_code=status_code,
                )

                # Retry only on server errors (5xx).
                if 500 <= status_code < 600 and attempt < _MAX_RETRIES:
                    logger.warning(
                        "Chamber API returned %d; retrying in %.1f s "
                        "(attempt %d/%d): %s",
                        status_code, backoff, attempt, _MAX_RETRIES,
                        error_message,
                    )
                    time.sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER
                    continue

                # Non-retryable or final attempt -- raise immediately.
                raise last_error from exc

            except urllib.error.URLError as exc:
                last_error = ChamberAPIError(
                    message=f"Network error contacting Chamber API: {exc.reason}",
                )
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Network error; retrying in %.1f s (attempt %d/%d): %s",
                        backoff, attempt, _MAX_RETRIES, exc.reason,
                    )
                    time.sleep(backoff)
                    backoff *= _BACKOFF_MULTIPLIER
                    continue
                raise last_error from exc

            except Exception as exc:
                raise ChamberAPIError(
                    message=f"Unexpected error during Chamber API request: {exc}",
                ) from exc

        # Should be unreachable, but satisfies the type checker.
        assert last_error is not None
        raise last_error

    @staticmethod
    def _handle_response(response: Any) -> dict:
        """Parse a successful HTTP response as JSON.

        Parameters
        ----------
        response:
            An ``http.client.HTTPResponse``-like object returned by
            ``urllib.request.urlopen``.

        Returns
        -------
        dict
            The parsed JSON response body.

        Raises
        ------
        ChamberAPIError
            If the response body cannot be parsed as JSON.
        """
        raw = response.read()
        if not raw:
            return {}

        charset = response.headers.get_content_charset() or "utf-8"
        body_text = raw.decode(charset)

        try:
            parsed = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise ChamberAPIError(
                message=(
                    f"Chamber API returned invalid JSON: {exc}. "
                    f"Body (first 200 chars): {body_text[:200]}"
                ),
                status_code=getattr(response, "status", 0),
            ) from exc

        # If the API wraps the result in {"data": ...}, unwrap it.
        if isinstance(parsed, dict) and "data" in parsed and len(parsed) <= 2:
            return parsed["data"]  # type: ignore[no-any-return]

        return parsed  # type: ignore[no-any-return]

    @staticmethod
    def _parse_error_message(error_body: str, status_code: int) -> str:
        """Extract a human-readable error message from an error response.

        Tries to parse the body as JSON and extract common error fields.
        Falls back to a generic message based on the status code.

        Parameters
        ----------
        error_body:
            Raw response body text from the error response.
        status_code:
            HTTP status code.

        Returns
        -------
        str
            A descriptive error message.
        """
        # Try JSON first.
        if error_body:
            try:
                parsed = json.loads(error_body)
                if isinstance(parsed, dict):
                    # Common error envelope shapes.
                    for key in ("message", "error", "detail", "error_description"):
                        value = parsed.get(key)
                        if isinstance(value, str) and value:
                            return value
                        if isinstance(value, dict):
                            inner = value.get("message", "")
                            if inner:
                                return str(inner)
            except (json.JSONDecodeError, AttributeError):
                pass

            # Fall back to raw body if it looks informative.
            stripped = error_body.strip()
            if stripped and len(stripped) < 500:
                return stripped

        # Generic messages for well-known status codes.
        generic_messages: Dict[int, str] = {
            400: "Bad request. Check the request payload.",
            401: "Authentication failed. Verify your Chamber API key.",
            403: "Access denied. Your API key may not have the required permissions.",
            404: "Resource not found. Verify the profile or job ID.",
            409: "Conflict. The resource may already exist.",
            422: "Validation error. Check the request payload.",
            429: "Rate limit exceeded. Please retry after a short delay.",
            500: "Internal server error. The Chamber service may be experiencing issues.",
            502: "Bad gateway. The Chamber service may be temporarily unavailable.",
            503: "Service unavailable. The Chamber service is under maintenance.",
            504: "Gateway timeout. The request took too long to process.",
        }
        return generic_messages.get(
            status_code,
            f"HTTP {status_code} error from Chamber API.",
        )
