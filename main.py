import argparse
import os
import torch.distributed as dist
import torch
import time

def main():
    parser = argparse.ArgumentParser("Mono Depth Estimation Codebase")
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        help="train or test "
                        )
    parser.add_argument("--exp_name_pre",
                        type=str,
                        default="NYU_L1Loss",
                        help="prename of experiment"
                        )
    parser.add_argument("--encoder",
                        type=str,
                        default="efficientlite3",
                        help="encoder backbone"
                        )
    #Dataset 
    parser.add_argument("--dataset",
                        type=str,
                        default="NYU",
                        help="dataset name"
                        )
    parser.add_argument("--dataset_path",
                        type=str,
                        default="data_utils/nyud.json",
                        help="the path of dataset"
                        )
    parser.add_argument("--input_height",
                        type=int,
                        default=480,
                        help="the height of model input"
                        )
    parser.add_argument("--input_width",
                        type=int,
                        default=640,
                        help="the width of model input"
                        )
    parser.add_argument("--max_depth",
                        type=int,
                        default=10,
                        help="the maxdepth of abs depth"
                        )
    
    # Log and save 
    parser.add_argument("--log_dir",
                        type=str,
                        default="./logs",
                        help="the directory of log"
                        )
    parser.add_argument("--log_freq",
                        type=int,
                        default=1000,
                        help="the frequency of log"
                        )
    parser.add_argument("--save_freq",
                        type=int,
                        default=10000,
                        help="the frequency of save model"
                        )
    parser.add_argument("--ckpt_name",
                        type=str,
                        default="",
                        help="the checkpoint name to save"
                        )
    parser.add_argument("--ckpt_path",
                        type=str,
                        default="",
                        help="path to a ckpt to load"
                        )
    
    #Training
    parser.add_argument("--scheduler",
                        type=str,
                        default="Poly",
                        help="scheduler type"
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="the batch size of training"
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="the learning rate of training"
                        )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-5,
                        help="weight decay of training"
                        )
    parser.add_argument("--adam_eps",
                        type=float,
                        default=1e-3,
                        help="the eps of adam optimizer"
                        )
    parser.add_argument("--retrain",
                        help="if used with checkpoint_path, will restart training from step zero",
                        action="store_true"
                        )
    parser.add_argument("--gamma",
                        type=float,
                        default=0.9,
                        help="learning rate decay"
                        )
    parser.add_argument("--fix_encoder_blocks",
                        help="if set, will fix the encoder blocks",
                        action="store_true"
                        )
    parser.add_argument("--bn_no_track_stats",
                        help="if set, will not track running stats in bn layers",
                        action="store_true"
                        )
    
    #Multi-gpu training
    parser.add_argument("--num_threads",
                        type=int,
                        default=3,
                        help="number of threads to use for data loading"
                        )
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="number of nodes for distributed training"
                        )
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="node rank for distributed training"
                        )
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="process rank for distributed training"
                        )
    parser.add_argument("--gpu_id",
                        type=int,
                        default=0,
                        help="if not ddp set the gpu_id")
    parser.add_argument("--ddp",
                        help="Use multi-processing distributed training to launch",
                        action="store_true"
                        )
    
    #Online eval
    parser.add_argument("--do_online_eval",
                        type=bool,
                        default=False,
                        help="the flag of online eval: True or False"
                        )
    parser.add_argument("--data_path_eval",
                        type=str,
                        default="",
                        help="pathe of the data for online evaluation")
    parser.add_argument("--eval_freq",
                        type=int,
                        default=500,
                        help="the frequency of eval")
    parser.add_argument("--min_depth_eval",
                        type=float,
                        default=1e-3,
                        help="the min eval value of abs depth"
                        )
    parser.add_argument("--max_depth_eval",
                        type=float,
                        default=10.0,
                        help="the max eval value of abs depth"
                        )
    parser.add_argument("--eval_summary_dir",
                        type=str,
                        default="",
                        help="eval summary directory"
                        )
    
    args = parser.parse_args()

    try:
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("gloo")
    except KeyError:
        rank = 0
        local_rank = 0
        world_size = 1
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )
    
    device = torch.device(f"cuda:{args.local_rank}")
    abs_depth_eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

if __name__ == "__main__":
    main()
