#! /bin/bash

for i in 0 ; do
    python test/image_predict.py ./test/test_parameters.yml > ./age_collector"$i".txt
    grep "age:" ./age_collector"$i".txt | cut -d' ' -f2 > ./age_collector"$i".txt.bak
    mv ./age_collector"$i".txt.bak ./age_collector"$i".txt
    paste -d' ' result.txt ./age_collector"$i".txt > result.txt.bak
    mv result.txt.bak result.txt
    rm ./age_collector"$i".txt
done
