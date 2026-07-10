"""
Loader for CVE data from the NVD (National Vulnerability Database) API v2.0.

Note: this queries a live public API — not reachable from Claude's sandboxed
tool environment, so this file is written and reviewed but validated by you
when you run it locally. It's a plain requests call, nothing exotic.

NVD rate limits: 5 requests / 30s without a key, 50 requests / 30s with a free
key (register at https://nvd.nist.gov/developers/request-an-api-key).
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests

from app.core.config import NVD_API_BASE, NVD_API_KEY, RAW_CVE_DIR


@dataclass
class CVERecord:
    cve_id: str
    description: str
    cvss_score: Optional[float]
    cvss_severity: Optional[str]
    published_date: str
    last_modified: str
    affected_products: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_document_text(self) -> str:
        lines = [
            f"{self.cve_id}",
            f"Severity: {self.cvss_severity or 'unspecified'} "
            f"(CVSS {self.cvss_score if self.cvss_score is not None else 'n/a'})",
            f"Published: {self.published_date}",
            "",
            self.description,
        ]
        if self.affected_products:
            lines += ["", "Affected products:", *[f"- {p}" for p in self.affected_products[:20]]]
        return "\n".join(lines)


def _parse_cve_item(item: dict) -> CVERecord:
    cve = item["cve"]
    cve_id = cve["id"]

    descriptions = cve.get("descriptions", [])
    description = next(
        (d["value"] for d in descriptions if d.get("lang") == "en"), ""
    )

    metrics = cve.get("metrics", {})
    cvss_score, cvss_severity = None, None
    # Prefer CVSS v3.1, fall back to v3.0, then v2 — NVD doesn't guarantee which are present
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            cvss_data = metrics[key][0]["cvssData"]
            cvss_score = cvss_data.get("baseScore")
            cvss_severity = cvss_data.get("baseSeverity", metrics[key][0].get("baseSeverity"))
            break

    affected_products = []
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                if match.get("vulnerable") and "criteria" in match:
                    affected_products.append(match["criteria"])

    references = [ref["url"] for ref in cve.get("references", [])]

    return CVERecord(
        cve_id=cve_id,
        description=description,
        cvss_score=cvss_score,
        cvss_severity=cvss_severity,
        published_date=cve.get("published", ""),
        last_modified=cve.get("lastModified", ""),
        affected_products=affected_products,
        references=references,
    )


def fetch_recent_cves(days_back: int = 30, results_per_page: int = 200, save: bool = True) -> list[CVERecord]:
    """
    Pull CVEs published in the last `days_back` days. For a prototype, start
    with a recent window (e.g. 30 days) rather than NVD's full ~250k-CVE
    history — you can widen the window once the pipeline works end to end.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    params = {
        "pubStartDate": start.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "pubEndDate": end.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "resultsPerPage": results_per_page,
        "startIndex": 0,
    }
    headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
    # Be polite to the free tier: 6s between requests if no key, ~0.7s with one.
    delay = 0.7 if NVD_API_KEY else 6.0

    all_records: list[CVERecord] = []
    all_raw_items: list[dict] = []

    while True:
        resp = requests.get(NVD_API_BASE, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        vulns = data.get("vulnerabilities", [])
        all_raw_items.extend(vulns)
        all_records.extend(_parse_cve_item(v) for v in vulns)

        total_results = data.get("totalResults", 0)
        params["startIndex"] += results_per_page
        if params["startIndex"] >= total_results:
            break
        time.sleep(delay)

    if save:
        RAW_CVE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_CVE_DIR / f"cves_{start.date()}_{end.date()}.json"
        out_path.write_text(json.dumps(all_raw_items))

    return all_records


if __name__ == "__main__":
    records = fetch_recent_cves(days_back=7)
    print(f"Fetched {len(records)} CVEs from the last 7 days")
    if records:
        print("\n--- Example record ---\n")
        print(records[0].to_document_text())
