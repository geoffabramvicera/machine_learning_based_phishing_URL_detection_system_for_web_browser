import csv
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit


class WebsiteReputationChecker:
    REQUIRED_COLUMNS = {
        "phish_id",
        "url",
        "phish_detail_url",
        "submission_time",
        "verified",
        "verification_time",
        "online",
        "target",
    }

    def __init__(self, csv_path, domain_resolver):
        self.csv_path = Path(csv_path)
        self.domain_resolver = domain_resolver

        self.loaded = False
        self.error = ""

        self.url_index = defaultdict(list)
        self.domain_index = defaultdict(list)

        self.record_count = 0

    @staticmethod
    def normalize_url(url):
        """Make a consistent URL key for exact report matching."""
        value = str(url or "").strip()

        if not value:
            return ""

        if "://" not in value:
            value = "http://" + value

        try:
            parts = urlsplit(value)
            hostname = (parts.hostname or "").lower().rstrip(".")

            if not hostname:
                return ""

            hostname = hostname.encode("idna").decode("ascii")

            port = parts.port

            if port in (80, 443):
                port = None

            host = hostname if port is None else f"{hostname}:{port}"
            path = parts.path or "/"

            # Scheme and fragment are ignored.
            # Hostname, path, and query are retained.
            return host + path + (
                "?" + parts.query if parts.query else ""
            )

        except (ValueError, UnicodeError):
            return ""

    @staticmethod
    def is_verified(value):
        return str(value or "").strip().lower() in {
            "yes", "true", "1", "verified"
        }

    def load_database(self):
        if self.loaded:
            return

        self.loaded = True

        if not self.csv_path.is_file():
            self.error = (
                f"Report database not found: {self.csv_path.name}"
            )
            return

        try:
            with self.csv_path.open(
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as csv_file:

                reader = csv.DictReader(csv_file)

                columns = set(reader.fieldnames or [])
                missing = self.REQUIRED_COLUMNS - columns

                if missing:
                    self.error = (
                        "CSV is missing columns: "
                        + ", ".join(sorted(missing))
                    )
                    return

                for row in reader:

                    # Only use verified phishing reports.
                    if not self.is_verified(row.get("verified")):
                        continue

                    original_url = (row.get("url") or "").strip()
                    url_key = self.normalize_url(original_url)

                    if not url_key:
                        continue

                    domain = self.domain_resolver(original_url)

                    if not domain:
                        continue

                    record = {
                        "phish_id": row.get("phish_id", ""),
                        "url": original_url,
                        "phish_detail_url": self.safe_report_link(
                            row.get("phish_detail_url", "")
                        ),
                        "submission_time": row.get(
                            "submission_time", ""
                        ),
                        "verified": row.get("verified", ""),
                        "verification_time": row.get(
                            "verification_time", ""
                        ),
                        "online": row.get("online", ""),
                        "target": row.get("target", "") or "Other",
                    }

                    self.url_index[url_key].append(record)
                    self.domain_index[domain].append(record)

                    self.record_count += 1

            # Display newer reports first.
            for records in self.url_index.values():
                records.sort(
                    key=lambda item: item["submission_time"],
                    reverse=True
                )

            for records in self.domain_index.values():
                records.sort(
                    key=lambda item: item["submission_time"],
                    reverse=True
                )

        except (OSError, UnicodeError, csv.Error) as error:
            self.error = "Could not load the report database: " + str(error)

            self.url_index.clear()
            self.domain_index.clear()
            self.record_count = 0

    @staticmethod
    def safe_report_link(value):
        """Only allow links to PhishTank report pages."""
        try:
            parts = urlsplit(value)
            host = (parts.hostname or "").lower()

            if (
                parts.scheme in {"http", "https"}
                and host in {
                    "phishtank.com",
                    "www.phishtank.com"
                }
            ):
                return value

        except ValueError:
            pass

        return ""

    def check(self, submitted_url):
        self.load_database()

        if self.error:
            return {
                "status": "database_unavailable",
                "message": self.error,
                "reports": [],
            }

        url_key = self.normalize_url(submitted_url)
        domain = self.domain_resolver(submitted_url)

        if not url_key or not domain:
            return {
                "status": "no_match",
                "message": "The URL could not be matched.",
                "reports": [],
            }

        exact_records = self.url_index.get(url_key, [])
        domain_records = self.domain_index.get(domain, [])

        if exact_records:
            status = "exact_match"
            message = (
                "This exact URL matches a verified "
                "PhishTank phishing report."
            )
            matched_records = exact_records

        elif domain_records:
            status = "related_domain_match"
            message = (
                "Other URLs on this registered domain have "
                "verified reports. The submitted URL itself "
                "is not an exact match."
            )
            matched_records = domain_records

        else:
            status = "no_match"
            message = (
                "No matching verified report was found "
                "in the local PhishTank database."
            )
            matched_records = []

        return {
            "status": status,
            "message": message,
            "registered_domain": domain,
            "exact_url_reported": bool(exact_records),
            "domain_has_reports": bool(domain_records),
            "total_domain_reports": len(domain_records),
            "records_loaded": self.record_count,
            "reports": matched_records[:5],
        }

    def health(self):
        self.load_database()

        return {
            "loaded": not bool(self.error),
            "records_loaded": self.record_count,
            "error": self.error,
        }