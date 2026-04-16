HF_MODEL=/path/to/hf/ckpt
OUT_DIR=/path/to/logs/dump

export HF_TOKEN=$(cat ~/hf_token)

lm_eval \
  --model hf \
  --model_args pretrained=${HF_MODEL},dtype=bfloat16 \
  --tasks commonsense_qa,openbookqa,arc_easy,arc_challenge,truthfulqa_mc1,truthfulqa_mc2,gsm8k,lambada_openai,hellaswag,piqa,winogrande,boolq,cb,copa,multirc,record,rte,wic,wsc,mmlu \
  --batch_size 128 \
  --device cuda:1 \
  --output_path ${OUT_DIR} \
  --trust_remote_code \
  --num_fewshot 0
