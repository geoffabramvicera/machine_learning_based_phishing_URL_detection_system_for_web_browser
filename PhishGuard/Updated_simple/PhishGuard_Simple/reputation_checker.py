"""Look up verified PhishTank reports in a local CSV file."""

import csv
import html
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class WebsiteReputationChecker:
    """Search a local PhishTank CSV by exact URL and registered domain."""

    REQUIRED_COLUMNS = {
        "url",
        "phish_detail_url",
        "submission_time",
        "verified",
        "verification_time",
        "online",
        "target",
    }

    def __init__(self, csv_path, domain_getter, max_results=5):
        self.csv_path = Path(csv_path)
        self.domain_getter = domain_getter
        self.max_results = max_results

        self.exact_index = defaultdict(list)
        self.domain_index = defaultdict(list)

        self.is_loaded = False
        self.records_loaded = 0
        self.error = ""

        self.load_data()

    @staticmethod
    def normalize_url(url):
        """Normalize URL casing, HTML entities, default ports, and fragments."""
        value = html.unescape(str(url or "")).strip()

        if not value:
            return ""

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
            value = "http://" + value

        try:
            parts = urlsplit(value)
            hostname = parts.hostname
            port = parts.port
        except ValueError:
            return ""

        if not hostname:
            return ""

        hostname = hostname.rstrip(".").lower()

        try:
            hostname = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            pass

        scheme = (parts.scheme or "http").lower()

        # Keep optional username/password information in the normalized URL.
        user_info = ""
        if parts.username is not None:
            user_info = parts.username
            if parts.password is not None:
                user_info += ":" + parts.password
            user_info += "@"

        # IPv6 hostnames need square brackets when rebuilding the URL.
        display_hostname = hostname
        if ":" in display_hostname and not display_hostname.startswith("["):
            display_hostname = "[" + display_hostname + "]"

        netloc = user_info + display_hostname

        # Remove only the standard port for the corresponding scheme.
        default_port = (scheme == "http" and port == 80) or (
            scheme == "https" and port == 443
        )
        if port is not None and not default_port:
            netloc += ":" + str(port)

        path = parts.path or "/"

        # Fragments are not sent to the web server, so they are not used
        # when comparing a submitted URL with a reported URL.
        return urlunsplit((scheme, netloc, path, parts.query, ""))

    @staticmethod
    def _safe_phishtank_link(value):
        """Allow only links that point to the expected PhishTank website."""
        value = html.unescape(str(value or "")).strip()

        try:
            parts = urlsplit(value)
        except ValueError:
            return ""

        hostname = (parts.hostname or "").lower()
        if hostname not in {"phishtank.com", "www.phishtank.com",
                            "phishtank.org", "www.phishtank.org"}:
            return ""

        if parts.scheme.lower() not in {"http", "https"}:
            return ""

        # Use HTTPS when opening a report page.
        return urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))

    def load_data(self):
        """Load verified reports and build URL/domain lookup indexes."""
        if not self.csv_path.exists():
            self.error = (
                "The reported-website CSV is missing. "
                "Place verified_online.csv beside app.py."
            )
            return

        try:
            with self.csv_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as csv_file:
                reader = csv.DictReader(csv_file)

                if not reader.fieldnames:
                    raise ValueError("The CSV file has no header row.")

                missing = self.REQUIRED_COLUMNS.difference(reader.fieldnames)
                if missing:
                    raise ValueError(
                        "The CSV is missing required columns: "
                        + ", ".join(sorted(missing))
                    )

                for row in reader:
                    # Only include records marked as verified.
                    if str(row.get("verified", "")).strip().lower() != "yes":
                        continue

                    reported_url = html.unescape(
                        str(row.get("url", "")).strip()
                    )
                    normalized_url = self.normalize_url(reported_url)

                    if not reported_url or not normalized_url:
                        continue

                    domain = self.domain_getter(reported_url)
                    if not domain:
                        continue

                    record = {
                        "phish_id": str(row.get("phish_id", "")).strip(),
                        "url": reported_url,
                        "phish_detail_url": self._safe_phishtank_link(
                            row.get("phish_detail_url", "")
                        ),
                        "submission_time": str(
                            row.get("submission_time", "")
                        ).strip(),
                        "verified": str(row.get("verified", "")).strip(),
                        "verification_time": str(
                            row.get("verification_time", "")
                        ).strip(),
                        "online": str(row.get("online", "")).strip(),
                        "target": html.unescape(
                            str(row.get("target", "")).strip()
                        ),
                    }

                    self.exact_index[normalized_url].append(record)
                    self.domain_index[domain].append(record)
                    self.records_loaded += 1

            # Newest reports appear first in the results.
            for records in self.exact_index.values():
                records.sort(
                    key=lambda item: item["submission_time"], reverse=True
                )

            for records in self.domain_index.values():
                records.sort(
                    key=lambda item: item["submission_time"], reverse=True
                )

            self.is_loaded = True
            self.error = ""

        except (OSError, csv.Error, ValueError) as error:
            self.is_loaded = False
            self.records_loaded = 0
            self.exact_index.clear()
            self.domain_index.clear()
            self.error = "Could not read the reported-website CSV: " + str(error)

    def check(self, submitted_url):
        """Return exact-URL or same-registered-domain report matches."""
        searched_domain = self.domain_getter(submitted_url)

        result = {
            "data_available": self.is_loaded,
            "source": "PhishTank",
            "dataset_records": self.records_loaded,
            "searched_domain": searched_domain,
            "match_type": "unavailable" if not self.is_loaded else "no_match",
            "total_matching_reports": 0,
            "reports": [],
            "message": "",
        }

        if not self.is_loaded:
            result["message"] = self.error
            return result

        normalized_url = self.normalize_url(submitted_url)
        exact_matches = self.exact_index.get(normalized_url, [])

        if exact_matches:
            match_type = "exact_url"
            matches = exact_matches
            result["message"] = (
                "This exact URL appears in the verified PhishTank dataset."
            )
        else:
            domain_matches = self.domain_index.get(searched_domain, [])
            if domain_matches:
                match_type = "same_registered_domain"
                matches = domain_matches
                result["message"] = (
                    "A reported URL exists on the same registered domain. "
                    "The submitted URL itself was not an exact match."
                )
            else:
                match_type = "no_match"
                matches = []
                result["message"] = (
                    "No matching report was found in this local dataset. "
                    "This does not guarantee that the website is safe."
                )

        result["match_type"] = match_type
        result["total_matching_reports"] = len(matches)
        result["reports"] = [
            dict(record, match_type=match_type)
            for record in matches[:self.max_results]
        ]

        return result
