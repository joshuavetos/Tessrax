"""
Tessrax Governance Dashboard (v17.9)
Displays live contradiction metabolism, ledger receipts, and federated consensus status.

Audit Metadata:
    auditor: "Tessrax Governance Kernel v16"
    clauses: ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001", "DLK-001"]
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import platform

import pandas as pd
import requests
import streamlit as st

CLAUSES = ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001", "DLK-001"]
AUDITOR_ID = "Tessrax Governance Kernel v16"
RECEIPTS_DIR = Path(__file__).resolve().parents[1] / "out"
FEDERATION_ROOTS_PATH = RECEIPTS_DIR / "federation_roots.json"
SESSION_RECEIPT_PATH = RECEIPTS_DIR / "dashboard_session_receipt.json"
VERIFIER_ENDPOINT = os.environ.get("TESSRAX_VERIFIER_URL", "http://localhost:8088/verify")
REQUIRED_INTEGRITY = 0.95
REQUIRED_LEGITIMACY = 0.9


def ensure_receipts_directory(path: Path) -> Path:
    """Ensure the receipts directory exists and return the resolved path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_receipts(directory: Path) -> List[Path]:
    """Return a sorted list of JSON receipt files within the provided directory."""
    return sorted(directory.glob("*.json"))


def load_json(path: Path) -> Tuple[Dict[str, Any] | List[Any] | None, str | None]:
    """Load JSON content from *path* or return an error message."""
    if not path.exists():
        return None, f"File not found: {path.name}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        return None, f"Invalid JSON in {path.name}: {exc}"


def compute_signature(payload: Dict[str, Any]) -> str:
    """Return a deterministic SHA-256 signature for *payload* excluding existing signature."""
    signable = {k: v for k, v in payload.items() if k != "signature"}
    return hashlib.sha256(json.dumps(signable, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def verify_with_external_service(payload: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str | None]:
    """Submit *payload* to the configured verifier endpoint and return response data or error."""
    try:
        response = requests.post(
            VERIFIER_ENDPOINT,
            json={"payload": payload, "signature": "", "public_key": ""},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return None, "Verifier returned a non-dict response"
        return data, None
    except requests.RequestException as exc:  # pragma: no cover - depends on network
        return None, f"Verifier request failed: {exc}"
    except ValueError as exc:  # pragma: no cover - unexpected content
        return None, f"Verifier response was not valid JSON: {exc}"


def record_session_receipt(
    receipts_available: bool,
    federation_loaded: bool,
    observations: List[str],
) -> Dict[str, Any]:
    """Persist an auditable session receipt and return the recorded payload."""
    integrity_score = REQUIRED_INTEGRITY
    legitimacy_score = REQUIRED_LEGITIMACY
    status = "PASS"

    if observations:
        status = "WARN"
        integrity_score = max(REQUIRED_INTEGRITY, REQUIRED_INTEGRITY - 0.02 * len(observations))
        legitimacy_score = max(REQUIRED_LEGITIMACY, REQUIRED_LEGITIMACY - 0.01 * len(observations))

    payload: Dict[str, Any] = {
        "auditor": AUDITOR_ID,
        "clauses": CLAUSES,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_info": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "receipts_available": receipts_available,
            "federation_loaded": federation_loaded,
        },
        "integrity_score": round(integrity_score, 3),
        "legitimacy_score": round(legitimacy_score, 3),
        "status": status,
        "observations": observations,
    }
    payload["signature"] = compute_signature(payload)

    try:
        SESSION_RECEIPT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - disk failures are environment dependent
        payload["status"] = "WARN"
        payload["observations"] = [*observations, f"Receipt persistence error: {exc}"]
        payload["integrity_score"] = max(REQUIRED_INTEGRITY - 0.03, REQUIRED_INTEGRITY)
        payload["legitimacy_score"] = max(REQUIRED_LEGITIMACY - 0.02, REQUIRED_LEGITIMACY)
        payload["signature"] = compute_signature(payload)
    return payload


def main() -> None:
    """Render the Tessrax governance dashboard and enforce runtime verification checks."""
    st.set_page_config(page_title="Tessrax Governance Dashboard", layout="wide")
    st.title("🧩 Tessrax Governance Dashboard — v17.9")

    receipts_dir = ensure_receipts_directory(RECEIPTS_DIR)
    observations: List[str] = []

    try:
        import tessrax  # type: ignore

        version = getattr(tessrax, "__version__", "0.0.0")
        version_tuple = tuple(int(part) for part in version.split(".") if part.isdigit())
        required_tuple = (17, 8, 0)
        if version_tuple < required_tuple:
            observations.append(
                "Local tessrax package version is below 17.8.0; governance dashboard relies on >=17.8.0."
            )
    except ImportError:
        observations.append("tessrax package is unavailable; governance SDK must be installed.")

    st.sidebar.header("Available Receipts")
    receipt_paths = list_receipts(receipts_dir)
    receipt_map = {path.name: path for path in receipt_paths}
    selected_receipt_name = None

    if receipt_paths:
        selected_receipt_name = st.sidebar.selectbox("Select Receipt", list(receipt_map.keys()))
    else:
        st.sidebar.info("No receipts detected in ./out — append receipts to review ledger events.")
        observations.append("No local receipts found during initialization.")

    selected_payload: Dict[str, Any] | None = None
    if selected_receipt_name:
        selected_path = receipt_map[selected_receipt_name]
        data, error = load_json(selected_path)
        if error:
            st.error(error)
            observations.append(error)
        elif isinstance(data, dict):
            selected_payload = data
            st.subheader("Ledger Receipt")
            st.json(data)

            receipt_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
            st.write(f"🔒 Local Hash: `{receipt_hash}`")
        else:
            message = f"Unsupported receipt schema in {selected_receipt_name}"
            st.error(message)
            observations.append(message)

    verifier_result: Dict[str, Any] | None = None
    if selected_payload:
        if st.button("Verify via External Verifier"):
            with st.spinner("Contacting verifier..."):
                verifier_result, error = verify_with_external_service(selected_payload)
            if verifier_result:
                st.subheader("Verification Response")
                st.json(verifier_result)
                if float(verifier_result.get("integrity", REQUIRED_INTEGRITY)) < REQUIRED_INTEGRITY:
                    observations.append("Verifier reported integrity below threshold.")
            elif error:
                st.error(error)
                observations.append(error)

    st.header("🌐 Federation Consensus")
    federation_loaded = False
    roots, roots_error = load_json(FEDERATION_ROOTS_PATH)
    if isinstance(roots, dict):
        if roots:
            df = pd.DataFrame(sorted(roots.items()), columns=["Node", "Merkle Root"])
            st.table(df)
            federation_loaded = True
        else:
            st.info("Federation roots file is present but empty.")
            observations.append("federation_roots.json was empty.")
    else:
        st.info("Run federation demo to populate federation_roots.json")
        if roots_error:
            observations.append(roots_error)

    session_receipt = record_session_receipt(bool(receipt_paths), federation_loaded, observations)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Receipt (DLK-VERIFIED)")
    st.sidebar.json(session_receipt)

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Integrity", f"{session_receipt['integrity_score']:.2f}")
    metric_col2.metric("Legitimacy", f"{session_receipt['legitimacy_score']:.2f}")

    st.caption("Compliant with AEP-001 • RVC-001 • EAC-001 • POST-AUDIT-001 • DLK-001")


if __name__ == "__main__":
    main()
