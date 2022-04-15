#!/bin/bash

overlap=(0.0 0.5)
coeff=(40 34 26 18)
segment=(5 4 3 2 1)
# segment=(5 4)
bases=(spotify_20)
augment=5
# bases=(base_portuguese_20)
# bases=(base_portuguese_4 base_portuguese_20)

# python3 /src/tcc_netro/main.py -c 40 -s 3 -o 0 -a 5 -b spotify_20

for l in ${!bases[@]}; do
    for k in ${!segment[@]}; do
        for i in ${!overlap[@]}; do
            for j in ${!coeff[@]}; do
                # echo "python3 /src/tcc/main.py -c ${coeff[$j]} -s ${segment[$k]} -o ${overlap[$i]} -a 30 -b ${bases[$l]}"
                python3 /src/tcc_netro/main.py -c ${coeff[$j]} -s ${segment[$k]} -o ${overlap[$i]} -a $augment -b ${bases[$l]}
            done
        done
    done
done


# python3 /src/tcc/inference.py -m /src/tcc/models/base_portuguese_20/SEG_1_OVERLAP_0_AUG_30/MFCC_18/D60_DO0_D0_DO0_D0/1649338837_86.86131238937378

# for w in ${!bases[@]}; do
#     PARENT_FOLDER="/src/tcc/models/${bases[$w]}"

#     find $PARENT_FOLDER -name inference -exec rm -rf {} \;

#     OUTPUT=$(python3 /src/tcc/backlog/run_folder.py -f $PARENT_FOLDER)

#     for i in $OUTPUT;
#     do
#         python3 /src/tcc/inference.py -m "$PARENT_FOLDER/$i" -i "/src/tcc/dataset/inference/${bases[$w]}"
#     done
# done