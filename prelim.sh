layers="0"
task="cola"
alpha="False"

save_path="/rscratch/tpang/kinshuk/RpMKin/bert_ft/lay_norm_False/alpha_asc_$alpha/layers_$layers/task_$task/lr2e-5_epoch20_bs32/"

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python bertft.py \
    --savepath "$save_path" \
    --epochs 20 \
    --model_name bert-base-uncased \
    --task_name "$task" \
    --learning_rate "2e-5" \
    --seed 5 \
    --freeze_bert True \
    --num_layers "$layers" \
    --alpha_ascending "$alpha"