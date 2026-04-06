#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_suiteSparse_dataset"
K128_DIR="${RESULTS_DIR}/k128"

INPUT_CSV="${K128_DIR}/results_128.csv"
DETAIL_CSV="${K128_DIR}/results_128_bsmr_reordering_details.csv"
PAPER_CSV="${K128_DIR}/results_128_bsmr_reordering_details_max_mn_group_summary_paper.csv"
PAPER_TEX="${K128_DIR}/results_128_bsmr_reordering_details_max_mn_group_summary_paper.tex"
PAPER_PNG="${K128_DIR}/results_128_bsmr_reordering_details_max_mn_group_summary_paper.png"
PAPER_PDF="${K128_DIR}/results_128_bsmr_reordering_details_max_mn_group_summary_paper.pdf"

echo "[1/3] Extracting BSMR reordering details from ${INPUT_CSV}"
python3 "${SCRIPT_DIR}/extract_bsmr_reordering_details.py" \
  --csv "${INPUT_CSV}"

echo "[2/3] Summarizing break-even statistics by max(M,N)"
python3 "${SCRIPT_DIR}/summarize_bsmr_reordering_groups.py" \
  --csv "${DETAIL_CSV}" \
  --group-by max_mn \
  --paper-ready \
  --latex-ready

echo "[3/3] Rendering paper table image/PDF"
python3 "${SCRIPT_DIR}/render_csv_table_image.py" \
  --csv "${PAPER_CSV}" \
  --title ""

echo
echo "Generated files:"
echo "  CSV : ${PAPER_CSV}"
echo "  TEX : ${PAPER_TEX}"
echo "  PNG : ${PAPER_PNG}"
echo "  PDF : ${PAPER_PDF}"
