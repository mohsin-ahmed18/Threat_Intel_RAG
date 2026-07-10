# Threat Intel RAG — progress log

## Done so far

- Project skeleton (`app/ingestion`, `app/retrieval`, `app/generation`, `app/core`, `app/api`)
- `app/core/config.py` — central config (paths, model names, API settings)
- `app/ingestion/attack_loader.py` — pulls the live MITRE ATT&CK STIX bundle,
  resolves technique-to-mitigation relationships, outputs clean `AttackTechnique`
  records. **Tested live**: parsed 697 techniques from the current bundle.
- `app/ingestion/cve_loader.py` — pulls CVEs from the NVD REST API v2.0 for a
  given date window, outputs clean `CVERecord` records. **Not yet run** —
  NVD isn't reachable from the build sandbox; run it yourself, see below.

## Running the CVE loader yourself

```bash
pip install -r requirements.txt
python -m app.ingestion.cve_loader
```

This fetches the last 7 days of published CVEs (edit `days_back` in the
`if __name__` block, or import `fetch_recent_cves(days_back=N)` from a script).

Optional but recommended: get a free NVD API key at
https://nvd.nist.gov/developers/request-an-api-key and set it before running:

```bash
export NVD_API_KEY=your-key-here
```

Without a key you're limited to 5 requests/30s (the loader already paces
itself at 6s between page requests to respect this). With a key it's
50 requests/30s.

## Next steps (not yet built)

1. Threat report / advisory loader (PDF + HTML)
2. Chunking — structure-aware for CVE/ATT&CK records, heading-aware for reports
3. Embedding pipeline (BGE-M3) + Qdrant collection setup
4. Retrieval + reranking
5. FastAPI query endpoint + LLM generation
