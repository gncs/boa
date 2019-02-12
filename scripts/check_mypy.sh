#!/usr/bin/env bash

# Check if all required Python type annotations are present

output_file="mypy.out"

mypy --config-file=".mypy.ini" setup.py boa > ${output_file} 2>&1

if [[ -n $(cat ${output_file}) ]] ; then
    echo "mypy found errors - please review the following code:"
    cat ${output_file}
    exit 1
fi