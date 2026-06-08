# OpenSanctions — attribution & license

Sanctions / PEP / adverse-media screening data is sourced from **OpenSanctions**.

- Source: https://www.opensanctions.org/ · bulk export https://data.opensanctions.org/datasets/
- Snapshot version: `TODO — pin on ingest day (e.g. 2026-06-06)`
- Collection: `default` (sanctions + PEP + crime/adverse-media-flagged entities)
- Format: FollowTheMoney JSON-lines (`*.jsonl`)
- License: **CC-BY-NC 4.0** (non-commercial use only)

## Compliance

This hackathon use is **non-commercial** (educational/competition), within the
CC-BY-NC terms. OpenSanctions is attributed here, in the submission slides, and in
the README. If TCS flags the NC clause post-hoc, swap to the OFAC SDN list (public
domain) — `ingest.py`'s entity normalization is schema-compatible.

The raw `*.jsonl` snapshot is git-ignored (large); fetch it during prep and place
it here as `snapshot.jsonl`, then run `server/ingest.py`.
