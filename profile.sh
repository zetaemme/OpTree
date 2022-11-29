#!/bin/bash
set -xe

python3 -m cProfile -o /tmp/opsion.prof main.py
snakeviz /tmp/opsion.prof