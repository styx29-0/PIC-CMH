import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from model import MyModel
import torch.optim as optim
from load_data import get_data
from utils import calc_map_k

loss_l2 = torch.nn.MSELoss()
criterion = nn.CosineSimilarity(dim=1)

def train_model(args, log, input_data_par, dataloader, task_index, result_dict):
    lr_set = np.linspace(args.lr, args.end_lr, args.epoch*(args.num_tasks)+1)
    learning_rate = np.linspace(lr_set[args.epoch*task_index], lr_set[args.epoch*(task_index+1)], args.epoch+1)
    # Initialize the hash layer
    hash_model = MyModel(args, task_index).cuda()
    hash_model.eval()
    # optimizer
    optimizer = optim.Adam(hash_model.parameters(), lr=learning_rate[0])
    # If it's not the first task then the previous model needs to be loaded
    if task_index != 0:
        hash_model.load(args)
        hash_model.cuda()
        for index_prompt in range(task_index):
            for ind in range(args.num_prompt):
                hash_model.expert_prompt_image_pool[index_prompt][ind] = hash_model.expert_prompt_image_pool[index_prompt][ind].cuda()
                hash_model.expert_prompt_text_pool[index_prompt][ind] = hash_model.expert_prompt_text_pool[index_prompt][ind].cuda()
    
    expert_prompt_image_list = []
    expert_prompt_text_list = []
    torch.manual_seed(args.seed)
    for _ in range(args.num_prompt):
        expert_prompt_image_list.append(nn.Parameter(torch.randn(1, args.feature_dim).clone().detach().requires_grad_(True)).cuda())
        expert_prompt_text_list.append(nn.Parameter(torch.randn(1, args.feature_dim).clone().detach().requires_grad_(True)).cuda())
    hash_model.expert_prompt_image_pool[task_index] = expert_prompt_image_list
    hash_model.expert_prompt_text_pool[task_index] = expert_prompt_text_list

    max_mapi2t = max_mapt2i = torch.zeros(1, dtype=torch.float32)
    max_mapi2t = max_mapi2t.cuda()
    max_mapt2i = max_mapt2i.cuda()
    train_dataloader = dataloader['train']
    for epoch in tqdm(range(args.epoch)):
        for batch in train_dataloader:
            image, text, label = batch
            image = image.cuda()
            text = text.cuda()
            label = label.cuda()
            hash_model.train()
            optimizer.zero_grad()

            # get hash code
            image_hash, text_hash = hash_model(args, image, text, task_index, separation=True)
            hash_similarity = image_hash.mm(text_hash.t())
            similarity_matrix = label.mm(label.t()) / 16.0
            similarity_matrix = 1 / (1+torch.exp(-similarity_matrix))
            similarity_matrix = 2 * similarity_matrix.float() - 1
            
            # 连续哈希码跨模态损失
            cross_loss = torch.sum(1 - criterion(image_hash, text_hash)) * args.alpha_param
            # 离散哈希码跨模态损失
            cross_sign_loss = torch.sum(1 - criterion(torch.sign(image_hash), torch.sign(text_hash))) * args.beta_param
            # 相似性保持损失
            sim_main = loss_l2(hash_similarity, similarity_matrix) * args.delta_param
            # 量化损失
            quantified_loss = (loss_l2(image_hash, torch.sign(image_hash)) + loss_l2(text_hash, torch.sign(text_hash))) * args.gamma_param
            ##########
            loss = sim_main + cross_loss + cross_sign_loss + quantified_loss

            # 优化
            loss.backward()
            optimizer.step()

        if epoch % args.valid_epoch == 0:
            if args.valid:
                hash_model.eval()
                with torch.no_grad():
                    separation = True
                    mapi2t_1000, mapt2i_1000 = valid_fun(hash_model, args,
                                                   input_data_par['test_image'], input_data_par['database_image'],
                                                   input_data_par['test_text'], input_data_par['database_text'],
                                                   input_data_par['test_label'], input_data_par['database_label'],
                                                   task_index, separation, top_k=1000)
                log.info('...epoch: %3d, valid MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (epoch + 1, mapi2t_1000, mapt2i_1000))
                if mapi2t_1000+mapt2i_1000 > max_mapi2t+max_mapt2i:
                    max_mapi2t = mapi2t_1000
                    max_mapt2i = mapt2i_1000
                    hash_model.save(args)
        hash_model.train()
            
        lr = learning_rate[epoch + 1]
        for param in optimizer.param_groups:
            param['lr'] = lr

    hash_model.load(args)
    hash_model.cuda()
    hash_model.eval()

    for i in range(task_index+1):
        _, _, _, test_image, test_text, test_label, database_image, database_text, database_label = get_data(args.data_dir, i, args.dataset)
        test_image = test_image.cuda()
        test_text = test_text.cuda()
        test_label = test_label.cuda()
        database_image = database_image.cuda()
        database_text = database_text.cuda()
        database_label = database_label.cuda()
        with torch.no_grad():
            separation = True
            mapi2t_1000, mapt2i_1000 = valid_fun(hash_model, args,
                                                    test_image, database_image,
                                                    test_text, database_text,
                                                    test_label, database_label,
                                                    task_index, separation, top_k=1000)
        log.info('...The {} data test is finished...'.format(i+1))
        log.info('...test MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (mapi2t_1000, mapt2i_1000))
        if task_index == 0:
            result_dict['valid_1000_i2t'][task_index, args.num_tasks+1] = mapi2t_1000.cpu().numpy()
            result_dict['valid_1000_t2i'][task_index, args.num_tasks+1] = mapt2i_1000.cpu().numpy()
        result_dict['valid_1000_i2t'][task_index, i] = mapi2t_1000.cpu().numpy()
        result_dict['valid_1000_t2i'][task_index, i] = mapt2i_1000.cpu().numpy()


def valid_fun(hash_model, args, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L, task_index, separation, top_k):
    qBX, qBY = generate_code(hash_model, query_x, query_y, args, task_index, separation)
    rBX, rBY = generate_code(hash_model, retrieval_x, retrieval_y, args, task_index, separation)

    mapi2t_1000 = calc_map_k(qBX, rBY, query_L, retrieval_L, top_k)
    mapt2i_1000 = calc_map_k(qBY, rBX, query_L, retrieval_L, top_k)
    return mapi2t_1000, mapt2i_1000


def generate_code(hash_model, X, Y, args, task_index, separation):
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B_X = torch.zeros(num_data, args.bit, dtype=torch.float).cuda()
    B_Y = torch.zeros(num_data, args.bit, dtype=torch.float).cuda()
    for i in range(num_data // args.batch_size + 1):
        ind = index[i * args.batch_size: min((i + 1) * args.batch_size, num_data)]
        image = X[ind].type(torch.float).cuda()
        text = Y[ind].type(torch.float).cuda()
        X_hash, Y_hash = hash_model(args, image, text, task_index, separation)
        B_X[ind] = X_hash
        B_Y[ind] = Y_hash
    B_X = torch.sign(B_X)
    B_Y = torch.sign(B_Y)
    return B_X, B_Y