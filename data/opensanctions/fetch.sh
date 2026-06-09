#!/usr/bin/env bash
# fetch.sh — download the OpenSanctions snapshot used by the screening agent.
#
# Defaults to a small, India-focused subset that's ~310 MB raw and gives
# realistic recall for the demo personas (Priya / Rajesh / Viktor):
#
#   in_sansad         19,079  Lok Sabha + Rajya Sabha members (PEPs)
#   in_nse_debarred   31,391  NSE-debarred entities (adverse media / fraud)
#   in_mha_banned        260  MHA-banned organizations (sanctions)
#   sanctions        283,621  UN / OFAC / EU / UK consolidated sanctions
#                   ─────────
#                    334,351  total
#
# All four are downloadable without auth from data.opensanctions.org.
#
# Usage:
#   ./fetch.sh                       # India subset, latest snapshot
#   ./fetch.sh --pin 20260610        # pin to a specific snapshot date
#   ./fetch.sh --full                # full default collection (~4.5 GB, 4.9M entities)
#   ./fetch.sh --ingest local        # also run ingest.py --backend local (fastembed)
#   ./fetch.sh --ingest vllm         # also run ingest.py --backend vllm (needs BGE on :8002)
#   ./fetch.sh --force               # re-download even if files exist
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

PIN="latest"
FULL=0
FORCE=0
INGEST=""

while [ $# -gt 0 ]; do
  case "$1" in
    --pin)     PIN="$2"; shift 2 ;;
    --full)    FULL=1; shift ;;
    --force)   FORCE=1; shift ;;
    --ingest)  INGEST="$2"; shift 2 ;;
    -h|--help) awk '/^[^#!]/{exit} /^# /{sub(/^# ?/,""); print}' "$0"; exit 0 ;;
    *)         echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

BASE="https://data.opensanctions.org/datasets/$PIN"
OUT="$HERE/snapshot.jsonl"

# ── Download ────────────────────────────────────────────────────────────────

fetch() {
  local slug="$1" file="$HERE/parts/$slug.jsonl"
  mkdir -p "$HERE/parts"
  if [ -s "$file" ] && [ "$FORCE" = "0" ]; then
    printf "    [skip]   %-20s %s lines (use --force to re-download)\n" \
      "$slug" "$(wc -l < "$file" | tr -d ' ')"
    return
  fi
  printf "    [fetch]  %-20s %s/%s/entities.ftm.json\n" "$slug" "$BASE" "$slug"
  curl -fL --progress-bar "$BASE/$slug/entities.ftm.json" -o "$file"
}

if [ "$FULL" = "1" ]; then
  echo "==> Fetching FULL default collection (snapshot=$PIN, ~4.5 GB, ~4.9M entities)"
  echo "    NOT recommended for the India demo — recall hits arbitrary global"
  echo "    designees before anything India-relevant. Continue in 5s; ^C to abort."
  sleep 5
  if [ -s "$OUT" ] && [ "$FORCE" = "0" ]; then
    echo "    [skip]   snapshot.jsonl already present ($(wc -l < "$OUT") lines)"
  else
    curl -fL --progress-bar "$BASE/default/entities.ftm.json" -o "$OUT"
  fi
else
  echo "==> Fetching India-focused subset (snapshot=$PIN)"
  fetch in_sansad
  fetch in_nse_debarred
  fetch in_mha_banned
  fetch sanctions
  echo "==> Concatenating parts → snapshot.jsonl"
  cat "$HERE/parts/"*.jsonl > "$OUT"
fi

ROWS=$(wc -l < "$OUT" | tr -d ' ')
SIZE=$(du -h "$OUT" | cut -f1)
echo "==> Done. $ROWS entities → $OUT ($SIZE)"

# ── Ingest (optional) ───────────────────────────────────────────────────────

if [ -z "$INGEST" ]; then
  echo
  echo "Next: ingest into server/opensanctions.db"
  echo "  Laptop (fastembed, no GPU):"
  echo "    cd server && python ingest.py --input ../data/opensanctions/snapshot.jsonl --backend local"
  echo "  AMD box (vLLM BGE on :8002):"
  echo "    cd server && python ingest.py --input ../data/opensanctions/snapshot.jsonl"
  echo
  echo "Or rerun this script with --ingest local | --ingest vllm to do it for you."
  exit 0
fi

case "$INGEST" in
  local|vllm) ;;
  *) echo "--ingest must be 'local' or 'vllm', got: $INGEST" >&2; exit 2 ;;
esac

echo "==> Ingesting with --backend $INGEST"
cd "$REPO/server"
python ingest.py --input "$OUT" --backend "$INGEST"
echo "==> Ingest complete. opensanctions.db is ready."
