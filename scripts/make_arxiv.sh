#!/bin/zsh
# Builds a flat, self-contained arXiv submission in paper/arxiv/ and zips it.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=paper/arxiv
rm -rf "$OUT" && mkdir -p "$OUT"
sed -e 's#\\input{../results/macros.tex}#\\input{macros.tex}#' \
    -e 's#../figures/##g' -e 's#../screenshots/##g' \
    paper/orch-neural-network.tex > "$OUT/orch-neural-network.tex"
cp results/macros.tex figures/fig_latency.pdf figures/fig_fidelity.pdf "$OUT/"
cp screenshots/iphone-6.9-0{1,2,3,4}-*.png "$OUT/"
( cd "$OUT" && latexmk -pdf -interaction=nonstopmode orch-neural-network.tex > build.log 2>&1 \
    && cp orch-neural-network.bbl . 2>/dev/null || true
  latexmk -c > /dev/null 2>&1 || true
  rm -f build.log
  zip -q -r ../orch-neural-network-arxiv.zip . -x '*.pdf' )
mv paper/orch-neural-network-arxiv.zip "$OUT/"
echo "arXiv package: $OUT/orch-neural-network-arxiv.zip"; ls -la "$OUT"
