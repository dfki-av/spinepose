#!/usr/bin/env bash
set -euo pipefail

# configs=(
#     "configs/spinepose-s_32xb256-10e_spinetrack-256x192.py"
#     "configs/spinepose-m_32xb256-10e_spinetrack-256x192.py"
#     "configs/spinepose-l_32xb256-10e_spinetrack-256x192.py"
#     "configs/spinepose-x_32xb128-10e_spinetrack-384x288.py"
# )

# checkpoints=(
#     "data/checkpoints/spinepose/spinepose-s_32xb256-10e_spinetrack-256x192.pth"
#     "data/checkpoints/spinepose/spinepose-m_32xb256-10e_spinetrack-256x192.pth"
#     "data/checkpoints/spinepose/spinepose-l_32xb256-10e_spinetrack-256x192.pth"
#     "data/checkpoints/spinepose/spinepose-x_32xb128-10e_spinetrack-384x288.pth"
# )

# for i in "${!configs[@]}"; do
#     python tools/test.py "${configs[i]}" "${checkpoints[i]}"
# done

python tools/summarize_results.py
