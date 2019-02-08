#!/usr/bin/env bash

# Apply YAPF recursively and inplace to all relevant files
# To be executed from the root directory

yapf --in-place --recursive --style=".style.yapf" setup.py aladdin_bo
