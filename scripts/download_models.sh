#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/checkpoints/spinepose
cd data/checkpoints/spinepose

wget https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-s_32xb256-10e_spinetrack-256x192.pth -O spinepose-s_32xb256-10e_spinetrack-256x192.pth
wget https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-m_32xb256-10e_spinetrack-256x192.pth -O spinepose-m_32xb256-10e_spinetrack-256x192.pth
wget https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-l_32xb256-10e_spinetrack-256x192.pth -O spinepose-l_32xb256-10e_spinetrack-256x192.pth
wget https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-x_32xb128-10e_spinetrack-384x288.pth -O spinepose-x_32xb128-10e_spinetrack-384x288.pth

cd ../../../
echo "Downloaded spinepose checkpoints to data/checkpoints/spinepose"
