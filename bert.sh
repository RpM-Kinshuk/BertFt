num_layers = "0 1 2 3 4 5 6 8 10 12 18 24 30 36 72 74"
task_list = "cola mnli mrpc qnli qqp rte sst2 stsb wnli"
alpha_list = "True False"


for alpha in $alpha_list
do
    for layers in $num_layers
    do
        for task in $task_list
        do
            save_path = "/RpMKin/data/bert_ft/lay_norm_False/alpha_asc_$alpha/layers_$num_layers/task_$task/lr2e-5_epoch20_bs32/"
            python bertft.py \
                --savepath $save_path \
                --epochs 20 \
                --model_name bert-base-uncased \
                --task_name $task \
                --max_length 512 \
                --batch_size 32 \
                --learning_rate '2e-5' \
                --seed 5 \
                --freeze_bert true \
                --num_layers $layers \
                --alpha_ascending $alpha \
                --slow_tokenizer True \
                --pad_to_max_length False \
                --add_layer_norm False \
                --max_train_steps 1000 \
                --grad_acc_steps 1 \
                --accelerate True
        done
    done
done