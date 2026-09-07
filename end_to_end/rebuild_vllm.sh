#!/bin/bash
# Shim dir so nvcc picks conda GCC 13 instead of system GCC 8.
mkdir -p /tmp/gcc-shim
ln -sf /home/pengliang/.conda/envs/vptq/bin/x86_64-conda-linux-gnu-gcc /tmp/gcc-shim/gcc
ln -sf /home/pengliang/.conda/envs/vptq/bin/x86_64-conda-linux-gnu-g++ /tmp/gcc-shim/g++
ln -sf /home/pengliang/.conda/envs/vptq/bin/x86_64-conda-linux-gnu-gcc /tmp/gcc-shim/cc
ln -sf /home/pengliang/.conda/envs/vptq/bin/x86_64-conda-linux-gnu-g++ /tmp/gcc-shim/c++

export PATH=/tmp/gcc-shim:/home/pengliang/.conda/envs/vptq/bin:$PATH
cd /home/pengliang/vllm-fpquant/build/temp.linux-x86_64-cpython-310 || exit 1
ninja -j 8 > /tmp/ninja_build.log 2>&1
echo "ninja exit: $?"
tail -40 /tmp/ninja_build.log
