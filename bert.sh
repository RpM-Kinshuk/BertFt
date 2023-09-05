num_layers = "0 1 2 3 4 5 6 8 10 12 18 24 30 36 72 74"
task_list = "cola mnli mrpc qnli qqp rte sst2 stsb wnli"

for task in $task_list
do
    for layers in $num_layers
    do
        python bertft.py \
            --savepath /models \
            --epochs 20 \
            --model_name bert-base-uncased \
            --task_name $task \
            --max_length 512 \
            --batch_size 32 \
            --learning_rate 2e-5 \
            --seed 5 \
            --freeze_bert True \
            --num_layers $layers \
            --alpha_ascending False \
            --slow_tokenizer True \
            --pad_to_max_length False \
            --add_layer_norm False \
            --max_train_steps 1000 \
            --grad_acc_steps 1 \
            --accelerate True \
    done
done