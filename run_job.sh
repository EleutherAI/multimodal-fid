#!/bin/bash

i=0
cat "$1" | while read; do
    echo "$REPLY"
    python ./vqgan.py "$REPLY" "$2" $i
    let "i+=1"
done