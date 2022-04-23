#!/bin/bash

overlap=(0 0.5)
coeff=(40 34 26 18)
segment=(5 4 3 2 1)
# segment=(5 4)
# bases=(spotify_20)
# bases=(spotify_60)
bases=(spotify_120)
augment=5
# bases=(base_portuguese_20)
# bases=(base_portuguese_4 base_portuguese_20)

# python3 /src/tcc_netro/main.py -c 40 -s 3 -o 0 -a 5 -b spotify_20

for l in ${!bases[@]}; do
    for k in ${!segment[@]}; do
        for i in ${!overlap[@]}; do
            for j in ${!coeff[@]}; do
                python3 /src/tcc_netro/main.py -c ${coeff[$j]} -s ${segment[$k]} -o ${overlap[$i]} -a $augment -b ${bases[$l]}
            done
        done
    done
done


# python3 /src/tcc/inference.py -m /src/tcc/models/base_portuguese_20/SEG_1_OVERLAP_0_AUG_30/MFCC_18/D60_DO0_D0_DO0_D0/1649338837_86.86131238937378

# for w in ${!bases[@]}; do
#     PARENT_FOLDER="/src/tcc_netro/models/${bases[$w]}"

#     find $PARENT_FOLDER -name inference -exec rm -rf {} \;

#     OUTPUT=$(python3 /src/tcc_netro/backlog/run_folder.py -f $PARENT_FOLDER)

#     for i in $OUTPUT;
#     do
#         CUDA_VISIBLE_DEVICES=1 python3 /src/tcc_netro/inference.py -m "$PARENT_FOLDER/$i" -i "/src/tcc_netro/dataset/inference_20/the weekend after hours.mp3" -c 7
#         CUDA_VISIBLE_DEVICES=1 python3 /src/tcc_netro/inference.py -m "$PARENT_FOLDER/$i" -i "/src/tcc_netro/dataset/inference_20/distrito 23 - invejoso.mp3" -c 4
#         CUDA_VISIBLE_DEVICES=1 python3 /src/tcc_netro/inference.py -m "$PARENT_FOLDER/$i" -i "/src/tcc_netro/dataset/inference_20/the weekend - blinding light.mp3" -c 5
#         CUDA_VISIBLE_DEVICES=1 python3 /src/tcc_netro/inference.py -m "$PARENT_FOLDER/$i" -i "/src/tcc_netro/dataset/inference_20/sevyn streeter - been a minute.mp3" -c 14
#         CUDA_VISIBLE_DEVICES=1 python3 /src/tcc_netro/inference.py -m "$PARENT_FOLDER/$i" -i "/src/tcc_netro/dataset/inference_20/Skan - No Glory.mp3" -c 12
#         CUDA_VISIBLE_DEVICES=1 python3 /src/tcc_netro/inference.py -m "$PARENT_FOLDER/$i" -i "/src/tcc_netro/dataset/inference_20/L7nnon - Sei que tu gosta muito.mp3" -c 19
#         CUDA_VISIBLE_DEVICES=1 python3 /src/tcc_netro/inference.py -m "$PARENT_FOLDER/$i" -i "/src/tcc_netro/dataset/inference_20/Baynk - Esther.mp3" -c 3
#     done
# done