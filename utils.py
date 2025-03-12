import os
import torch
import random
import logging
import numpy as np
from torch.nn import functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def logger(args):
    '''
    '\033[0;34m%s\033[0m': blue
    :return:
    '''
    logger = logging.getLogger('Prompt')
    logger.setLevel(logging.DEBUG)
    log_floder_path = './logs/'
    os.makedirs(log_floder_path, exist_ok=True)
    
    txt_log = logging.FileHandler(log_floder_path + args.dataset + '_' + str(args.bit) + '.log')

    txt_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    print(f'log will be stored to {txt_log}')

    return logger


def checkpoints(args):
    checkpoints_floder_path = './checkpoints/'
    os.makedirs(checkpoints_floder_path, exist_ok=True)
    args.checkpoints_dir = os.path.join(checkpoints_floder_path, args.dataset + '_' + str(args.bit))
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    print(f'checkpoints will be stored under {args.checkpoints_dir}')


def creat_result_dict(args):
    I2T = np.zeros((args.num_tasks, args.num_tasks+1))
    T2I = np.zeros((args.num_tasks, args.num_tasks+1))
    result_dict = {
        'I2T' : I2T,
        'T2I' : T2I,
    }
    return result_dict


def save_result_dict(args, result_dict):
    # Average the data
    for i in range(args.num_tasks):
        map_sum_I2T = 0
        map_sum_T2I = 0
        for j in range(i+1):
            map_sum_I2T += result_dict['I2T'][i, j]
            map_sum_T2I += result_dict['T2I'][i, j]
        result_dict['I2T'][i, args.num_tasks] = map_sum_I2T / (i+1)
        result_dict['T2I'][i, args.num_tasks] = map_sum_T2I / (i+1)

    # Specify the CSV filepath
    csv_floder = './result/'
    os.makedirs(csv_floder, exist_ok=True)
    csv_file = csv_floder + '{}_{}.csv'.format(args.dataset, args.bit)

    # Write the numpy matrix from the dictionary to a CSV file
    with open(csv_file, 'w') as f:
        for key, matrix in result_dict.items():
            # Write the data for each matrix to a file
            f.write(key + '\n')
            np.savetxt(f, matrix, delimiter=',', fmt='%f')
    print(f'The result is already stored under {csv_file}')


def train_select_prompt(args, sample, expert_prompt_pool, task_index):
    prompt = torch.cat(expert_prompt_pool[task_index], dim=0)
    prompt = F.normalize(prompt)
    select_prompt_sim_matrix = sample @ prompt.t()
    max_indices = torch.argmax(select_prompt_sim_matrix, dim=1)
    select_prompt = [expert_prompt_pool[task_index][q] for q in max_indices]
    select_prompt = torch.cat(select_prompt, dim=0)
    return select_prompt


def valid_select_prompt(args, sample, expert_prompt_pool, task_index):
    prompt = F.normalize(torch.cat([torch.cat(expert_prompt_pool[i], dim=0) for i in range(task_index+1)])).cuda()
    select_prompt_sim_matrix = sample @ prompt.t()
    max_indices = torch.argmax(select_prompt_sim_matrix, dim=1)
    max_indices = max_indices.cuda()
    select_prompt = []
    for q in max_indices:
        if (q/args.num_prompt).floor().to(torch.int) == 0:
            select_prompt.append(expert_prompt_pool[0][q%args.num_prompt])
        elif (q/args.num_prompt).floor().to(torch.int) == 1:
            select_prompt.append(expert_prompt_pool[1][q%args.num_prompt])
        elif (q/args.num_prompt).floor().to(torch.int) == 2:
            select_prompt.append(expert_prompt_pool[2][q%args.num_prompt])
        elif (q/args.num_prompt).floor().to(torch.int) == 3:
            select_prompt.append(expert_prompt_pool[3][q%args.num_prompt])
        else:
            select_prompt.append(expert_prompt_pool[4][q%args.num_prompt])
    select_prompt = torch.cat(select_prompt, dim=0)
    return select_prompt


def calc_map_k(qu_B, re_B, qu_L, re_L, topk=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = qu_L.shape[0]
    map = 0
    if topk is None:
        topk = re_L.shape[0]
    for iter in range(num_query):
        q_L = qu_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(re_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qu_B[iter, :], re_B)
        _, ind = torch.sort(hamm, stable=True)   # 默认稳定排序
        ind.squeeze_()
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = torch.sum(tgnd)
        if tsum == 0:
            continue
        count = torch.arange(1, int(tsum) + 1).type(torch.float32)
        tindex = torch.nonzero(tgnd).squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_neighbor(label1, label2, device):
    # calculate the similar matrix
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    Sim = Sim.to(device)
    return Sim