num_layers="0 1 2 3 4 5 6 8 10 12 18 24 30 36 72 74"
task_list="cola mnli mrpc qnli qqp rte sst2 stsb wnli"
alpha_list="True False"
laynorm="False"

for task in $task_list
do
    for alpha in $alpha_list
    do
        for layers in $num_layers
        do
            save_path="/rscratch/tpang/kinshuk/RpMKin/bert_ft/task_$task/lay_norm_$laynorm/alpha_asc_$alpha/layers_$num_layers/lr2e-5_epoch3_bs32/"
            python bertft.py \
                --savepath "$save_path" \
                --epochs 3 \
                --model_name bert-base-uncased \
                --task_name "$task" \
                --max_length 128 \
                --batch_size 32 \
                --learning_rate "2e-5" \
                --seed 7 \
                --freeze_bert True \
                --num_layers "$layers" \
                --alpha_ascending "$alpha" \
                --slow_tokenizer True \
                --pad_to_max_length True \
                --add_layer_norm $laynorm \
                --max_train_steps 1000 \
                --grad_acc_steps 1 \
                --accelerate False \
                --debug False
        done
    done
done