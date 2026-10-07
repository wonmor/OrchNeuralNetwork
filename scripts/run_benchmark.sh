#!/bin/zsh
# Runs BenchmarkTests on a simulator (default) or a connected device and saves the JSON.
#   scripts/run_benchmark.sh                       -> results/bench_simulator.json
#   scripts/run_benchmark.sh "id=<device-udid>" results/bench_iphone17.json
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-"platform=iOS Simulator,name=iPhone 18 Pro"}
OUT=${2:-results/bench_simulator.json}
LOG=$(mktemp)
xcodebuild test -project OrchNeuralNetwork.xcodeproj -scheme OrchNeuralNetwork \
  -destination "$DEST" -only-testing:OrchNeuralNetworkTests/BenchmarkTests \
  -derivedDataPath build/DerivedData 2>&1 | tee "$LOG" | grep -E "Test Case|error:|ORCH_BENCH_DONE|\*\* TEST" || true
grep -m1 "ORCH_BENCH_JSON" "$LOG" | sed 's/.*ORCH_BENCH_JSON //' | python3 -m json.tool > "$OUT"
echo "wrote $OUT"
