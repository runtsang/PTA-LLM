![image](https://github.com/runtsang/PTA-LLM/blob/master/imgs/ptallm.jpg)

**Official implementation of NeurIPS "Probabilistic Token Alignment for Large Language Model Fusion"**

**Contact me:** [runjia.tech](https://runjia.tech/) | rz4545@rit.edu | runjia@msn.com

[Paper](https://arxiv.org/abs/2509.17276) | [Homepage](https://runjia.tech/neurips_pta-llm/)

# 📣News

(👉Under construction! There are several redundancies in the current version, and the commands/instructions are not perfectly ready for formal release. I will gradually update it! Please stay tuned.)

**2024/09/22:** Our [homepage](https://runjia.tech/neurips_pta-llm/) is available now (slides and video are on the way)! Check it out to see more details.

**2025/09/21:** Our code is publicly available now! Thank you for your attention and patience!

#  📰Poster

![image](https://github.com/runtsang/PTA-LLM/blob/master/imgs/poster.jpg)

## Data Construction

We use the [MiniPile](https://huggingface.co/datasets/JeanKaddour/minipile) dataset for continual training. 

Here we show the scripts to obtain representations from multiple LLMs for model fusion.

1. Split long text

```bash
python ./src/utils/split_long_text.py \
  --base_model_name_or_path "<path_to_llama_2_7b>" \
  --blending_model_name_or_path "<path_to_open_llama_7b_v2>" \
  --another_blending_model_name_or_path "<path_to_mpt_7b>" \
  --dataset "<path_to_minipile>" \
  --dataset_save_dir "<path_to_minipile_split>" \
  --cache_dir "<path_to_cache_dir>" \
  --block_size 2048 \
  --preprocessing_num_workers 80
```

2. Get representations for each LLM

```bash
# We split the dataset into 8 splits, then process each split on a GPU.
# Please run this script for llama_2_7b, open_llama_7b_v2, and mpt_7b.
for i in {0..7}; do
export CUDA_VISIBLE_DEVICES=${i}
python ./src/utils/forward_for_logits.py \
  --model_name_or_path "<path_to_each_model>" \
  --dataset "<path_to_minipile_split>" \
  --dataset_save_dir "${i}_8_<path_to_minipile_split_each_model_representation>" \
  --dataset_split_num 8 \
  --dataset_index ${i} \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --training_mode full \
  --load_in_half bf16 \
  --batch_size 8 \
  --preprocessing_num_workers 80 \
  --top_k_logits 10 \
  --save_per_token_metric 2>&1 > "${i}_8_<path_to_log_file>" 2>&1 &
unset CUDA_VISIBLE_DEVICES
sleep 30
done

wait
```

![image](https://github.com/runtsang/PTA-LLM/blob/master/imgs/overview.jpg)

3. Align representations from different LLMs

```bash
# Get vocab mapping from different LLMs.

# llama_2_7b <-> open_llama_7b_v2
python ./src/utils/vocab_mapping.py \
  --base_model_name_or_path "<path_to_llama_2_7b>" \
  --blending_model_name_or_path "<path_to_open_llama_7b_v2>" \
  --dataset_dir "<path_to_minipile_split>" \
  --vocab_mapping_save_dir "<path_to_llama_2_7b_open_llama_7b_v2_vocab_mapping>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --vocab_mapping_type "default" \
  --num_process 1

# llama_2_7b <-> mpt_7b
python ./src/utils/vocab_mapping.py \
  --base_model_name_or_path "<path_to_llama_2_7b>" \
  --blending_model_name_or_path "<path_to_mpt_7b>" \
  --dataset_dir "<path_to_minipile_split>" \
  --vocab_mapping_save_dir "<path_to_llama_2_7b_mpt_7b_vocab_mapping>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --vocab_mapping_type "default" \
  --num_process 1
```

```bash
# Align representations from different LLMs.

# llama_2_7b <-> open_llama_7b_v2
for i in {0..7}; do
python ./src/utils/token_alignment_ot.py \
  --base_model_name_or_path "<path_to_llama_2_7b>" \
  --blending_model_name_or_path "<path_to_open_llama_7b_v2>" \
  --base_dataset_dir "${i}_8_<path_to_minipile_split_llama_2_7b_representation>" \
  --blending_dataset_dir "${i}_8_<path_to_minipile_split_open_llama_7b_v2_representation>" \
  --dataset_save_dir "${i}_8_<path_to_minipile_split_llama_2_7b_open_llama_7b_v2_aligned_representation>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --preprocessing_num_workers 80 \
  --batch_size 100 \
  --blending_model_index 0 \
  --vocab_align_type "soft" \
  --vocab_mapping_save_dir "<path_to_llama_2_7b_open_llama_7b_v2_vocab_mapping>" \
  --metric_level "sequence"
done 

# llama_2_7b <-> mpt_7b
for i in {0..7}; do
python ./src/utils/token_alignment_ot.py \
  --base_model_name_or_path "<path_to_llama_2_7b>" \
  --blending_model_name_or_path "<path_to_mpt_7b>" \
  --base_dataset_dir "${i}_8_<path_to_minipile_split_llama_2_7b_open_llama_7b_v2_aligned_representation>" \
  --blending_dataset_dir "${i}_8_<path_to_minipile_split_mpt_7b_representation>" \
  --dataset_save_dir "${i}_8_<path_to_minipile_split_llama_2_7b_open_llama_7b_v2_mpt_7b_aligned_representation>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --preprocessing_num_workers 80 \
  --batch_size 100 \
  --blending_model_index 1 \
  --vocab_align_type "soft" \
  --vocab_mapping_save_dir "<path_to_llama_2_7b_mpt_7b_vocab_mapping>" \
  --metric_level "sequence"
done
```

4. Packing all features to speed up training.

```bash
for i in {0..7}; do
python3 ./src/utils/packing.py \
  --dataset_dir "${i}_8_<path_to_minipile_split_llama_2_7b_open_llama_7b_v2_mpt_7b_aligned_representation>" \
  --dataset_save_dir "${i}_8_<path_to_miniplie_fusellm_processed>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --preprocessing_num_workers 80 \
  --batch_size 1000 \
  --metric_level "sequence"
```

The final processed data is at `${i}_8_<path_to_miniplie_fusellm_processed>`, where `i in {0..7}`.

## Training

Here, we show the script for PTA-LLM training.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed --master_port=20001 ./src/train.py \
  --training_mode full \
  --deepspeed ./config/zero_stage2_config.json \
  --model_name_or_path "<path_to_llama_2_7b>" \
  --output_dir "<path_to_save_fusellm_7b>" \
  --model_max_length 2048 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --evaluation_strategy steps \
  --per_device_eval_batch_size 1 \
  --logging_strategy steps \
  --do_train \
  --do_eval \
  --bf16 True \
  --tf32 True \
  --warmup_ratio 0.008 \
  --lr_scheduler_type cosine \
  --dataset_name "0_8_<path_to_miniplie_fusellm_processed>,1_8_<path_to_miniplie_fusellm_processed>,2_8_<path_to_miniplie_fusellm_processed>,3_8_<path_to_miniplie_fusellm_processed>,4_8_<path_to_miniplie_fusellm_processed>,5_8_<path_to_miniplie_fusellm_processed>,6_8_<path_to_miniplie_fusellm_processed>,7_8_<path_to_miniplie_fusellm_processed>" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --eval_steps 500 \
  --optim adamw_torch \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --seed 42 \
  --gradient_checkpointing True \
  --use_flash_attn True \
  --report_to tensorboard \
  --do_distill \
  --distill_with_ref_model True \
  --distill_with_aligned_model_0 True \
  --distill_with_aligned_model_1 True \
  --distill_loss_type "ce" \
  --distill_teacher_temperature 1.0 \
  --lm_loss_weight 0.9 \
  --distill_greater_as_gt True \
  --distill_greater_as_gt_type "hard" \
  --dataloader_num_workers 10 \
  --remove_unused_columns False 2>&1 | tee "<path_to_log_file>"
```

## Evaluation

We conducted our evaluations using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework.


## Citation

```
@inproceedings{zeng2025probabilistic,
  title={Probabilistic Token Alignment for Large Language Model Fusion},
  author={Zeng, Runjia and Liang, James Chenhao and Han, Cheng and Cao, Zhiwen and Liu, Jiahao and Quan, Xiaojun and Chen, Yingjie Victor and Huang, Lifu and Geng, Tong and Wang, Qifan and Liu, Dongfang},
  booktitle={NeurIPS},
  year={2025}
}
```

## Acknowledgments

The documentation above and code are copied and modified  from **[FuseAI](https://github.com/fanqiwan/FuseAI)**. Thanks for their effort.
