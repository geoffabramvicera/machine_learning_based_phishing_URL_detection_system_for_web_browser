"""PhishGuard, a simple local phishing URL detector."""

import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

import joblib
from flask import Flask, jsonify, request, send_from_directory
from scipy.sparse import hstack

from reputation_checker import WebsiteReputationChecker

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model.pkl"
SUFFIX_PATH = BASE_DIR / "public_suffix_list.dat"

SUSPICIOUS_WORDS = {
    "account", "activate", "auth", "bank", "billing", "bonus", "captcha",
    "confirm", "credential", "crypto", "gift", "invoice", "login", "logon",
    "payment", "password", "recover", "recovery", "secure", "security",
    "signin", "support", "unlock", "update", "validate", "verification",
    "verify", "wallet", "webscr", "session", "confirmidentity"
}

BRAND_NAMES = {
    "adobe", "amazon", "apple", "bankofamerica", "binance", "chase",
    "coinbase", "discord", "dropbox", "facebook", "github", "google",
    "instagram", "linkedin", "microsoft", "netflix", "office", "outlook",
    "paypal", "steam", "telegram", "tiktok", "twitter", "wellsfargo",
    "whatsapp", "yahoo", "youtube", "zoom"
}

KNOWN_MAJOR_DOMAINS = {
    "adobe.com", "amazon.com", "apple.com", "brave.com", "cloudflare.com",
    "duckduckgo.com", "facebook.com", "github.com", "google.com", "instagram.com",
    "linkedin.com", "microsoft.com", "mozilla.org", "netflix.com", "openai.com",
    "paypal.com", "python.org", "reddit.com", "stackoverflow.com", "wikipedia.org",
    "yahoo.com", "youtube.com", "zoom.us"
}

app = Flask(__name__)
model_bundle = None
model_error = "Model not trained yet. Run train_model.py first."


def load_suffix_list():
    """Read domain suffix rules used to find the registered domain."""
    exact_rules = set()
    wildcard_rules = set()
    exception_rules = set()

    if not SUFFIX_PATH.exists():
        return exact_rules, wildcard_rules, exception_rules

    with open(SUFFIX_PATH, "r", encoding="utf-8") as file:
        for line in file:
            rule = line.strip().lower()

            if not rule or rule.startswith("//"):
                continue
            if rule.startswith("!"):
                exception_rules.add(rule[1:])
            elif rule.startswith("*."):
                wildcard_rules.add(rule[2:])
            else:
                exact_rules.add(rule)

    return exact_rules, wildcard_rules, exception_rules


EXACT_SUFFIXES, WILDCARD_SUFFIXES, EXCEPTION_SUFFIXES = load_suffix_list()


def get_registered_domain(hostname):
    """Return the main registered domain, such as example.co.uk."""
    hostname = hostname.lower().strip(".")
    if not hostname:
        return ""

    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass

    labels = hostname.split(".")
    if len(labels) == 1:
        return hostname

    suffix_length = 1

    for start in range(len(labels)):
        suffix = ".".join(labels[start:])

        if suffix in EXCEPTION_SUFFIXES:
            suffix_length = len(labels) - start - 1
            break

        if suffix in EXACT_SUFFIXES:
            suffix_length = max(suffix_length, len(labels) - start)

        if start + 1 < len(labels):
            wildcard_tail = ".".join(labels[start + 1:])
            if wildcard_tail in WILDCARD_SUFFIXES:
                suffix_length = max(suffix_length, len(labels) - start)

    if len(labels) <= suffix_length:
        return hostname

    return ".".join(labels[-(suffix_length + 1):])


def parse_url(url):
    """Parse URLs from the dataset, even when a scheme is missing."""
    value = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        value = "http://" + value
    try:
        return urlsplit(value)
    except ValueError:
        return urlsplit("")


def get_group_domain(url):
    """Get the registered domain used to keep related URLs together."""
    parts = parse_url(url)
    hostname = (parts.hostname or "").lower()

    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        pass

    return get_registered_domain(hostname) or hostname or url




# Load the local reported-website CSV once when the backend starts.
REPORTED_WEBSITES_PATH = BASE_DIR / "verified_online.csv"
reputation_checker = WebsiteReputationChecker(
    REPORTED_WEBSITES_PATH, get_group_domain
)

def get_known_major_domain(url):
    """Check whether the exact hostname is on the small local domain list."""
    parts = parse_url(url)
    hostname = (parts.hostname or "").lower().rstrip(".")

    if parts.scheme.lower() not in ["http", "https"]:
        return None
    if parts.username or parts.password:
        return None

    try:
        port = parts.port
    except ValueError:
        return None

    expected_port = 443 if parts.scheme.lower() == "https" else 80
    if port not in [None, expected_port]:
        return None

    if hostname.startswith("www."):
        hostname = hostname[4:]

    if hostname in KNOWN_MAJOR_DOMAINS:
        return hostname

    return None


def is_known_homepage(url):
    """Recognize only the exact HTTPS homepage of a listed major domain."""
    parts = parse_url(url)

    if parts.scheme.lower() != "https":
        return None
    if parts.path not in ["", "/"] or parts.query or parts.fragment:
        return None

    return get_known_major_domain(url)


def extract_features(url):
    """Turn a URL into simple numeric values for the machine learning model."""
    parts = parse_url(url)
    hostname = (parts.hostname or "").lower()
    try:
        hostname_ascii = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        hostname_ascii = hostname

    registered_domain = get_registered_domain(hostname_ascii)
    path = parts.path or ""
    query = parts.query or ""
    labels = hostname_ascii.split(".") if hostname_ascii else []
    domain_labels = registered_domain.split(".") if registered_domain else []

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    suspicious_count = 0
    for word in SUSPICIOUS_WORDS:
        if word in url.lower():
            suspicious_count += 1

    brand_in_hostname = 0
    brand_in_subdomain = 0
    registered_label = domain_labels[0] if domain_labels else ""

    for brand in BRAND_NAMES:
        if brand in hostname_ascii:
            brand_in_hostname = 1
            if brand in hostname_ascii.replace(registered_domain, ""):
                brand_in_subdomain = 1

    brand_mismatch = 0
    for brand in BRAND_NAMES:
        if brand in registered_label and registered_label != brand:
            brand_mismatch = 1

    try:
        port = parts.port
    except ValueError:
        port = None

    standard_port = 443 if parts.scheme.lower() == "https" else 80
    has_nonstandard_port = int(port is not None and port != standard_port)

    path_parts = []
    for part in path.split("/"):
        if part:
            path_parts.append(part)

    query_parts = []
    for part in query.split("&"):
        if part:
            query_parts.append(part)
    file_extension = Path(path).suffix.lower()
    risky_extensions = {".exe", ".apk", ".bat", ".cmd", ".jar", ".msi", ".ps1", ".scr"}

    digit_count = 0
    for character in url:
        if character.isdigit():
            digit_count += 1

    return {
        "url_length": len(url),
        "hostname_length": len(hostname_ascii),
        "path_length": len(path),
        "query_length": len(query),
        "digit_count": digit_count,
        "hyphen_count": hostname.count("-"),
        "dot_count": hostname.count("."),
        "subdomain_count": max(0, len(labels) - len(domain_labels)),
        "path_depth": len(path_parts),
        "query_parameter_count": len(query_parts),
        "uses_https": int(parts.scheme.lower() == "https"),
        "has_ip_address": has_ip,
        "has_punycode": int("xn--" in hostname_ascii),
        "has_non_ascii_hostname": int(hostname != hostname_ascii),
        "has_at_sign": int("@" in url),
        "has_userinfo": int(parts.username is not None or parts.password is not None),
        "has_nonstandard_port": has_nonstandard_port,
        "suspicious_word_count": suspicious_count,
        "brand_in_hostname": brand_in_hostname,
        "brand_in_subdomain": brand_in_subdomain,
        "brand_mismatch": brand_mismatch,
        "has_suspicious_file": int(file_extension in risky_extensions),
    }


def get_url_text(url):
    """Return the hostname, path, and query for character-pattern analysis."""
    parts = parse_url(url)
    hostname = (parts.hostname or "").lower()
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        pass

    return " ".join([
        hostname,
        (parts.path or "").lower(),
        (parts.query or "").lower()
    ])


def validate_url(url):
    """Return an error message when the user enters an invalid URL."""
    if not isinstance(url, str) or not url.strip():
        return "Please enter a URL."
    if len(url) > 2048:
        return "The URL is too long."
    for character in url:
        if character.isspace():
            return "The URL must not contain spaces."

    try:
        parts = urlsplit(url)
        hostname = parts.hostname
        parts.port
    except ValueError:
        return "The URL format is invalid."

    if parts.scheme.lower() not in ["http", "https"]:
        return "Start the URL with http:// or https://."
    if not parts.netloc or not hostname:
        return "Enter a complete URL, such as https://example.com."

    return None


def load_model():
    """Load the trained model the first time it is needed."""
    global model_bundle, model_error

    if model_bundle is not None:
        return True

    try:
        loaded_model = joblib.load(MODEL_PATH)
        required_parts = ["model", "numeric_vectorizer", "url_vectorizer"]

        for part in required_parts:
            if part not in loaded_model:
                raise ValueError("This is an older model. Run train_model.py again.")

        model_bundle = loaded_model
        model_error = ""
        return True
    except FileNotFoundError:
        model_error = "Model file is missing. Run train_model.py first."
    except Exception as error:
        model_error = "Could not load the model: " + str(error)

    return False


@app.route("/")
def home():
    """Show the simple web page."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    ready = load_model()
    return jsonify({
        "model_loaded": ready,
        "error": model_error,
        "reputation_loaded": reputation_checker.is_loaded,
        "reported_records": reputation_checker.records_loaded,
        "reputation_error": reputation_checker.error
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not load_model():
        return jsonify({"error": model_error}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with a url field."}), 400

    url = data.get("url", "")
    if not isinstance(url, str):
        return jsonify({"error": "The URL must be text."}), 400
    url = url.strip()

    error = validate_url(url)
    if error:
        return jsonify({"error": error}), 400

    try:
        bundle = model_bundle
        features = extract_features(url)
        numeric_matrix = bundle["numeric_vectorizer"].transform([features])
        text_matrix = bundle["url_vectorizer"].transform([get_url_text(url)])
        full_matrix = hstack([numeric_matrix, text_matrix], format="csr")

        model = bundle["model"]
        prediction = int(model.predict(full_matrix)[0])
        probabilities = model.predict_proba(full_matrix)[0]
        phishing_index = list(model.classes_).index(1)
        phishing_score = round(float(probabilities[phishing_index]) * 100, 2)

        known_domain = get_known_major_domain(url)
        known_homepage = is_known_homepage(url)
        review_required = known_domain is not None and prediction == 1 and known_homepage is None

        if known_homepage:
            is_phishing = False
            result_label = "Recognized major-domain homepage"
        elif review_required:
            is_phishing = False
            result_label = "Known domain, but URL needs review"
        else:
            is_phishing = prediction == 1
            if is_phishing:
                result_label = "Potential phishing URL"
            else:
                result_label = "No phishing indicators detected"

        reputation_result = reputation_checker.check(url)

        return jsonify({
            "url": url,
            "reputation": reputation_result,
            "is_phishing": is_phishing,
            "phishing_score": phishing_score,
            "result_label": result_label,
            "review_required": review_required,
            "recognized_homepage": known_homepage is not None
        })
    except Exception:
        app.logger.exception("URL analysis failed")
        return jsonify({"error": "The URL could not be analyzed. Check the backend console."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
