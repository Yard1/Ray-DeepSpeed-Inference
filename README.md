# Ray-DeepSpeed-Inference

Based on https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/text-generation


```bash
python run_on_every_node.py "s3://large-dl-models-mirror/models--anyscale--opt-66b-resharded/main/" "~/model"

python deepspeed_inference_actors.py --name "facebook/opt-66b" --checkpoint_path "~/model" --batch_size 1 --ds_inference --use_kernel  --use_meta_tensor --num_worker_groups 1 --num_gpus_per_worker_group 12
```