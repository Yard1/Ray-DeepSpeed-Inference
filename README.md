# Ray-DeepSpeed-Inference

Based on https://github.com/microsoft/DeepSpeedExamples/tree/master/inference/huggingface/text-generation


```bash
python deepspeed_inference_actors.py --name facebook/opt-66b --batch_size 1 --ds_inference --use_kernel  --use_meta_tensor --reshard_checkpoint_path "/nvme/resharded_checkpoint" --hf_home "/nvme/cache" --num_worker_groups 1 --num_gpus_per_worker_group 12
```