#!/bin/bash

status=true

./gauss AB.csv out.csv

if [ -f out.csv ]; then
    echo "Output file created"
else
    echo "Output file not created"
    status=false
fi

$status
