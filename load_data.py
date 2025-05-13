import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    # 返回索引对应的图像文本和标签
    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index, :]
        label = self.labels[index, :]
        return img, text, label

    # 返回数据集长度
    def __len__(self):
        # sun1 = len(self.images)
        # sum2 = len(self.labels)
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count
    
    
def get_data(data_path, task_index, dataset):
    if dataset == 'MSCOCO' or dataset == 'MSCOCO_NoMean':
        train_image = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/train_{}_image.mat".format(task_index), 'r')['train_{}_image'.format(task_index)], 0, 1))
        train_text = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/train_{}_text.mat".format(task_index), 'r')['train_{}_text'.format(task_index)], 0, 1))
        train_label = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/train_{}_label.mat".format(task_index), 'r')['train_{}_lab'.format(task_index)], 0, 1))

        test_image = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/test_{}_image.mat".format(task_index), 'r')['test_{}_image'.format(task_index)], 0, 1))
        test_text = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/test_{}_text.mat".format(task_index), 'r')['test_{}_text'.format(task_index)], 0, 1))
        test_label = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/test_{}_label.mat".format(task_index), 'r')['test_{}_lab'.format(task_index)], 0, 1))

        database_image = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/database_{}_image.mat".format(task_index), 'r')['database_{}_image'.format(task_index)], 0, 1))
        database_text = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/database_{}_text.mat".format(task_index), 'r')['database_{}_text'.format(task_index)], 0, 1))
        database_label = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/database_{}_label.mat".format(task_index), 'r')['database_{}_lab'.format(task_index)], 0, 1))
    elif dataset == 'NUSWIDE' or dataset == 'NUSWIDE_NoMean':
        train_image = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/train_task_{}_image.mat".format(task_index), 'r')['train_task_{}_image'.format(task_index)], 0, 1))
        train_text = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/train_task_{}_text.mat".format(task_index), 'r')['train_task_{}_text'.format(task_index)], 0, 1))
        train_label = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/train_task_{}_label.mat".format(task_index), 'r')['train_task_{}_label'.format(task_index)], 0, 1))

        test_image = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/test_task_{}_image.mat".format(task_index), 'r')['test_task_{}_image'.format(task_index)], 0, 1))
        test_text = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/test_task_{}_text.mat".format(task_index), 'r')['test_task_{}_text'.format(task_index)], 0, 1))
        test_label = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/test_task_{}_label.mat".format(task_index), 'r')['test_task_{}_label'.format(task_index)], 0, 1))

        database_image = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/database_task_{}_image.mat".format(task_index), 'r')['database_task_{}_image'.format(task_index)], 0, 1))
        database_text = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/database_task_{}_text.mat".format(task_index), 'r')['database_task_{}_text'.format(task_index)], 0, 1))
        database_label = torch.from_numpy(np.swapaxes(h5py.File(data_path + "/database_task_{}_label.mat".format(task_index), 'r')['database_task_{}_label'.format(task_index)], 0, 1))
    return train_image, train_text, train_label, test_image, test_text, test_label, database_image, database_text, database_label

def get_loader(args, task_index):
    train_image, train_text, train_label, test_image, test_text, test_label, database_image, database_text, database_label = get_data(args.dataset_path, task_index, args.dataset)

    imgs = {'train': train_image, 'test': test_image, 'database': database_image}
    texts = {'train': train_text, 'test': test_text, 'database': database_text}
    labels = {'train': train_label, 'test': test_label, 'database': database_label}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['train', 'test', 'database']}

    shuffle = {'train': True, 'test': False, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=args.batch_size, shuffle=shuffle[x], num_workers=0) for x in ['train', 'test', 'database']}
    
    num_train = train_image.shape[0]
    img_dim = train_image.shape[1]
    text_dim = train_text.shape[1]
    num_class = train_label.shape[1]

    input_data_par = {}
    input_data_par['train_image'] = train_image.cuda()
    input_data_par['train_text'] = train_text.cuda()
    input_data_par['train_label'] = train_label.cuda()

    input_data_par['test_image'] = test_image.cuda()
    input_data_par['test_text'] = test_text.cuda()
    input_data_par['test_label'] = test_label.cuda()

    input_data_par['database_image'] = database_image.cuda()
    input_data_par['database_text'] = database_text.cuda()
    input_data_par['database_label'] = database_label.cuda()

    input_data_par['num_sample'] = num_train
    input_data_par['num_class'] = num_class

    return input_data_par, dataloader