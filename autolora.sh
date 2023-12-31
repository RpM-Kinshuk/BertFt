task_list="cola sst2"
seed=7
sby="autolora"
batch_size=32
epochs=3
order="True"
layers=10

for task in $task_list
do
    save_path="/rscratch/tpang/kinshuk/RpMKin/bert_ft/GLUE/trainseed_$seed/task_$task/lay_norm_False/$sby/lr2e-5_epoch3_bs$batch_size/"

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
    python /rscratch/tpang/kinshuk/RpMKin/bert_ft/bertft.py \
            --savepath $save_path \
            --epochs $epochs \
            --model_name bert-base-uncased \
            --task_name $task \
            --sortby $sby \
            --alpha_ascending $order \
            --batch_size $batch_size \
            --learning_rate "2e-5" \
            --seed $seed \
            --num_layers $layers \
            --verbose True \
            --debug False \
            --memlog True
done