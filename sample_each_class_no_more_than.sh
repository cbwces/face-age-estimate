#! /bin/bash

if [[ $# -eq 0 ]]; then
    echo "param1: file of input image paths with label"
    echo "param2: max number of images in each label"
else
    shuf "$1" > "$1".tmp
    deal_file="$1".tmp
    max_sample="$2"
    awk -v ms="$max_sample" '{if (lb[$2] < ms){lb[$2]+=1; print $0}}' "$deal_file"
    rm $deal_file
fi
exit
