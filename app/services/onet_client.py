"""O*NET Web Services API client."""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class ONetClientError(Exception):
    """Base exception for O*NET client errors."""
    pass


class ONetAuthError(ONetClientError):
    """Authentication failure."""
    pass


class ONetRateLimitError(ONetClientError):
    """Rate limit exceeded."""
    pass


class ONetTimeoutError(ONetClientError):
    """Request timeout."""
    pass


class ONetClient:
    """Client for O*NET Web Services API.

    Handles authentication, retries, rate limiting, and error handling.
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """Initialize O*NET client.

        Args:
            username: O*NET username (defaults to settings)
            password: O*NET password (defaults to settings)
            base_url: API base URL (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)
            max_retries: Maximum retry attempts (defaults to settings)
        """
        self.username = username or settings.onet_username
        self.password = password or settings.onet_password
        self.base_url = (base_url or settings.onet_base_url).rstrip("/")
        self.timeout = timeout or settings.onet_timeout
        self.max_retries = max_retries or settings.onet_max_retries

        # Check if credentials are available
        if not self.username or not self.password:
            logger.warning("O*NET credentials not configured. Client will fail on API calls.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to O*NET API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            ONetAuthError: Authentication failed
            ONetRateLimitError: Rate limit exceeded
            ONetTimeoutError: Request timeout
            ONetClientError: Other API errors
        """
        if not self.username or not self.password:
            raise ONetAuthError("O*NET credentials not configured")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    auth=(self.username, self.password),
                    headers={
                        "Accept": "application/json",
                        "User-Agent": f"{settings.app_name}/1.0",
                    },
                )

                # Handle different status codes
                if response.status_code == 401:
                    raise ONetAuthError("Authentication failed. Check credentials.")
                elif response.status_code == 429:
                    raise ONetRateLimitError("Rate limit exceeded. Retry later.")
                elif response.status_code >= 500:
                    raise ONetClientError(f"Server error: {response.status_code}")
                elif response.status_code >= 400:
                    raise ONetClientError(
                        f"Client error: {response.status_code} - {response.text}"
                    )

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as e:
                logger.error(f"Request timeout for {url}: {e}")
                raise ONetTimeoutError(f"Request timeout: {e}")
            except httpx.NetworkError as e:
                logger.error(f"Network error for {url}: {e}")
                raise ONetClientError(f"Network error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error for {url}: {e}")
                raise ONetClientError(f"Unexpected error: {e}")

    async def search_occupations(self, query: str, limit: int = 20) -> List[Dict[str, str]]:
        """Search for occupations by keyword.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of dicts with 'code' and 'title' keys

        Example:
            [
                {"code": "15-1252.00", "title": "Software Developers"},
                {"code": "15-1299.08", "title": "Web Developers"},
            ]
        """
        try:
            # O*NET search endpoint
            data = await self._request(
                "GET",
                f"/online/search",
                params={"keyword": query, "end": limit},
            )

            occupations = []
            if "occupation" in data:
                for occ in data["occupation"]:
                    occupations.append({
                        "code": occ.get("code", ""),
                        "title": occ.get("title", ""),
                    })

            logger.info(f"Found {len(occupations)} occupations for query: {query}")
            return occupations

        except ONetClientError:
            raise
        except Exception as e:
            logger.error(f"Error searching occupations: {e}")
            raise ONetClientError(f"Search failed: {e}")

    async def get_occupation_meta(self, onet_code: str) -> Dict[str, Any]:
        """Get occupation metadata.

        Args:
            onet_code: O*NET-SOC code (e.g., "15-1252.00")

        Returns:
            Dict with title, description, job_zone, education, etc.

        Example:
            {
                "code": "15-1252.00",
                "title": "Software Developers",
                "description": "Research, design, and develop...",
                "job_zone": 4,
                "education": "Bachelor's degree",
            }
        """
        try:
            # Get occupation details
            data = await self._request("GET", f"/online/occupations/{onet_code}")

            # Parse job zone
            job_zone = None
            if "job_zone" in data:
                try:
                    job_zone = int(data["job_zone"])
                except (ValueError, TypeError):
                    pass

            # Parse education
            education = None
            if "education" in data and isinstance(data["education"], list):
                if len(data["education"]) > 0:
                    education = data["education"][0].get("category", {}).get("title")

            result = {
                "code": onet_code,
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "job_zone": job_zone,
                "education": education,
                "raw_data": data,
            }

            logger.info(f"Fetched metadata for {onet_code}")
            return result

        except ONetClientError:
            raise
        except Exception as e:
            logger.error(f"Error fetching occupation metadata: {e}")
            raise ONetClientError(f"Failed to fetch metadata: {e}")

    async def get_occupation_skills(self, onet_code: str) -> List[Dict[str, Any]]:
        """Get skills for an occupation.

        Args:
            onet_code: O*NET-SOC code

        Returns:
            List of skill dicts with element_id, name, importance, level

        Example:
            [
                {
                    "element_id": "2.B.1.a",
                    "skill_name": "Reading Comprehension",
                    "importance": 75.0,
                    "level": 5.0,
                },
            ]
        """
        try:
            # Get skills for occupation
            data = await self._request(
                "GET",
                f"/online/occupations/{onet_code}/summary/skills"
            )

            skills = []
            if "element" in data and isinstance(data["element"], list):
                for elem in data["element"]:
                    skill_name = elem.get("name", "")
                    element_id = elem.get("id", "")

                    # Parse importance and level from scale items
                    importance = None
                    level = None

                    if "score" in elem and isinstance(elem["score"], list):
                        for score_item in elem["score"]:
                            scale_name = score_item.get("scale", {}).get("name", "").lower()
                            value = score_item.get("value")

                            if value is not None:
                                try:
                                    value = float(value)
                                except (ValueError, TypeError):
                                    continue

                                if "importance" in scale_name:
                                    importance = value
                                elif "level" in scale_name:
                                    level = value

                    if element_id and skill_name:
                        skills.append({
                            "element_id": element_id,
                            "skill_name": skill_name,
                            "importance": importance,
                            "level": level,
                        })

            logger.info(f"Fetched {len(skills)} skills for {onet_code}")
            return skills

        except ONetClientError:
            raise
        except Exception as e:
            logger.error(f"Error fetching occupation skills: {e}")
            raise ONetClientError(f"Failed to fetch skills: {e}")


class MockONetClient(ONetClient):
    """Mock O*NET client for demo mode."""

    # Demo data with realistic occupations and skills
    MOCK_OCCUPATIONS = {
        "15-1252.00": {
            "code": "15-1252.00",
            "title": "Software Developers",
            "description": "Research, design, and develop computer and network software or specialized utility programs.",
            "job_zone": 4,
            "education": "Bachelor's degree",
        },
        "15-1299.08": {
            "code": "15-1299.08",
            "title": "Web Developers",
            "description": "Develop and implement websites, web applications, application databases, and interactive web interfaces.",
            "job_zone": 3,
            "education": "Associate's degree",
        },
        "15-1244.00": {
            "code": "15-1244.00",
            "title": "Network and Computer Systems Administrators",
            "description": "Install, configure, and maintain an organization's local area network (LAN), wide area network (WAN), data communications network, operating systems, and physical and virtual servers.",
            "job_zone": 3,
            "education": "Bachelor's degree",
        },
    }

    MOCK_SKILLS = {
        "15-1252.00": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72.0, "level": 5.12},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 75.0, "level": 5.12},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 66.0, "level": 4.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81.0, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 84.0, "level": 5.88},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.8.e", "skill_name": "Systems Evaluation", "importance": 75.0, "level": 5.25},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 84.0, "level": 5.75},
        ],
        "15-1299.08": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 66.0, "level": 4.62},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 66.0, "level": 4.62},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66.0, "level": 4.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75.0, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 81.0, "level": 5.62},
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66.0, "level": 4.75},
        ],
        "15-1244.00": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0, "level": 5.00},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75.0, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81.0, "level": 5.50},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 72.0, "level": 5.12},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 84.0, "level": 5.75},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 66.0, "level": 4.50},
        ],
    }

    async def _request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Override to return mock data."""
        # Simulate network delay
        await asyncio.sleep(0.1)
        return {}

    async def search_occupations(self, query: str, limit: int = 20) -> List[Dict[str, str]]:
        """Return mock occupation search results."""
        await asyncio.sleep(0.1)
        query_lower = query.lower()

        results = []
        for code, data in self.MOCK_OCCUPATIONS.items():
            if query_lower in data["title"].lower():
                results.append({"code": code, "title": data["title"]})

        logger.info(f"[DEMO] Found {len(results)} occupations for query: {query}")
        return results[:limit]

    async def get_occupation_meta(self, onet_code: str) -> Dict[str, Any]:
        """Return mock occupation metadata."""
        await asyncio.sleep(0.1)

        if onet_code not in self.MOCK_OCCUPATIONS:
            raise ONetClientError(f"Occupation {onet_code} not found in demo data")

        data = self.MOCK_OCCUPATIONS[onet_code].copy()
        data["raw_data"] = data.copy()

        logger.info(f"[DEMO] Fetched metadata for {onet_code}")
        return data

    async def get_occupation_skills(self, onet_code: str) -> List[Dict[str, Any]]:
        """Return mock skills."""
        await asyncio.sleep(0.1)

        if onet_code not in self.MOCK_SKILLS:
            raise ONetClientError(f"Skills for {onet_code} not found in demo data")

        skills = self.MOCK_SKILLS[onet_code]
        logger.info(f"[DEMO] Fetched {len(skills)} skills for {onet_code}")
        return skills


def get_onet_client() -> ONetClient:
    """Factory function to get appropriate O*NET client.

    Returns:
        ONetClient or MockONetClient based on configuration
    """
    if settings.is_demo_mode:
        logger.info("Using MockONetClient (demo mode)")
        return MockONetClient()
    else:
        logger.info("Using ONetClient (live mode)")
        return ONetClient()
