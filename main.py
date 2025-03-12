import os
import yaml
import argparse
from train import train_model
from load_data import get_loader
from utils import logger, set_seed, checkpoints, creat_result_dict, save_result_dict
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.yaml", type=str)
    args = parser.parse_args()
    assert os.path.isfile(args.cfg), "cfg file: {} not found".format(args.cfg)

    # merge config.yaml
    with open(args.cfg, 'r') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    args_dict = vars(args)
    args_dict.update(yaml_config)
    args = argparse.Namespace(**args_dict)

    return args


def main():
    args = parse_args()
    args.dataset_path = os.path.join(args.data_path, args.dataset)
    checkpoints(args)
    log = logger(args)
    set_seed(args.seed)

    result_dict = creat_result_dict(args)
    
    for task_index in range(args.num_tasks):

        input_data_par, dataloader = get_loader(args, task_index=task_index)
        
        train_model(args, log, input_data_par, dataloader, task_index, result_dict)
        print(f'The {task_index+1} task is trained')

    save_result_dict(args, result_dict)

if __name__ == '__main__':
    main()