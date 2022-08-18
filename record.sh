#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Invalid syntax."
  echo "Syntax: record.sh <recorded_object_name> <out_file_name>"
else
  # Record data
  #python scripts/record.py -o huawei_box -f huawei_box -d 5 --fps 30
  python scripts/record.py -o $1 -f $2 -d 3 --fps 60
fi
