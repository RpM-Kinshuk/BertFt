# BertFt

This research project explores different methods of fine-tuning the pre-trained <I>Google BERT model</I> on various datasets from the <I>General Language Understanding Evaluation (GLUE)</I> benchmark.
<br> <br>
The project also highlights the variations in performance observed over different training layers of the model.
<br>
The results obtained thus far are new to the academia and have not been obtained before.
<br>
Model chosen to obtain the results is the <I>bert-base</I> model and the tokenizer used is the <I>BERT Tokenizer</I>  from <I>HuggingFace Transformers</I>.
<br><br>
Following are the choices of hyperparameters used: <br>
<b>learning_rate = 2e-5 <br>
batch_size = 32 <br>
epochs = 3 <br>
optimizer = <I>ADAMW</I> <br>
padding = max_length</b>

<h3> CoLA </h2>

<img src = "https://github.com/RpM-Kinshuk/BertFt/assets/103813028/4d7baaa4-dd42-4f74-bcb2-fa57c33c3085" width = 50%>
<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/ac2b0a89-5084-45b0-99b3-2763f00d81b7" width = 50%>

<h3> MRPC </h3>

<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/6bccb453-f650-411e-b13e-732fd8514043" width = 50%>
<img src = "https://github.com/RpM-Kinshuk/BertFt/assets/103813028/f6237f00-d0b3-4f98-862b-d0dcb3eca87e" width = 50%>


<h3> QNLI </h3>

<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/ee68d381-93af-4784-ad52-082561d669d3" width = 50%>
<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/b91204dc-1f8c-4961-bece-45c00e8b090b" width = 50%>


<h3> RTE </h3>

<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/f217c0b2-4a44-4578-8de5-3a72398cca61" width = 50%>
<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/46cfa5bf-685f-4fea-af3a-d5e6bb2feea4" width = 50%>


<h3> SST-2 </h3>

<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/a3f9d4ac-6b84-48bd-a9b3-511c29af3d6b" width = 50%>
<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/de913a51-bd4e-46ae-8293-13f12b3d64c5" width = 50%>

<h3> STSB </h3>

<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/626a1e79-d680-451e-a1d3-976a7d18c61e" width = 50%>
<img src="https://github.com/RpM-Kinshuk/BertFt/assets/103813028/19d79ae0-7d7a-4752-9def-8a2d75333997" width = 50%>

<br> <br>
Script usage:

num_layers="0 1 2 3 4 5 6 8 10 12 18 24 30 36 72 74"
task_list="mrpc qnli qqp rte sst2 stsb wnli"
alpha_list="True False"
laynorm="False"
model="bert-base-uncased"
batch_size=32

for task in $task_list <br>
do <br>
    for alpha in $alpha_list <br>
    do <br>
        for layers in $num_layers <br>
        do <br>
            save_path="YOUR_SAVE_PATH" <br>
            OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python bertft.py \
                --savepath "$save_path" \
                --epochs 3 \
                --model_name $model \
                --task_name "$task" \
                --max_length 128 \
                --batch_size $batch_size \
                --learning_rate "2e-5" \
                --seed 7 \
                --freeze True \
                --num_layers "$layers" \
                --alpha_ascending "$alpha" \
                --slow_tokenizer True \
                --pad_to_max_length True \
                --add_layer_norm $laynorm \
                --max_train_steps 1000 \
                --grad_acc_steps 1 \
                --accelerate False \
                --debug False <br>
        done <br>
    done <br>
done <br>
