#!/usr/bin/env bash

# Check if all Python code is formatted properly

output_file="yapf.out"

yapf --diff --recursive --style=".style.yapf" setup.py boa > ${output_file} 2>&1

if [[ -n $(cat ${output_file}) ]] ; then
    echo "yapf found errors - please review the following code:"
    cat ${output_file}
    exit 1
fi