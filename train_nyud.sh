CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=2 \
main.py \
--mode train \
--exp_name_pre NYUD_L1Loss \
--encoder efficientnet_lite3 \
--dataset NYUD \
--dataset_path data_utils/nyud_train.json \
--input_height 480 \
--input_width 640 \
--max_depth 10 \
--log_dir ./logs \
--log_freq 1000 \
--save_freq 10000 \
--ckpt_name abs_depth_iter_ \
--scheduler Poly \
--batch_size 4 \
--learning_rate 2e-5 \
--weight_decay 1e-5 \
--adam_eps 1e-3 \
--gamma 0.9 \
--num_threads 3 \
--gpu_id 0 \
--do_online_eval True \
--data_path_eval data_utils/nyud_test.json \
--eval_freq 500 \
--min_depth_eval 1e-3 \
--max_depth_eval 10.0 \
--eval_summary_dir eval_results \

