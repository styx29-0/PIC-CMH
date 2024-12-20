import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from utils import train_select_prompt, valid_select_prompt

class BasicModule(torch.nn.Module):
    """
    封装nn.Module，主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, args):
        checkpoints = torch.load(args.checkpoints_dir + 'hash_model.pth', map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoints['model_state_dict'])
        if self.training:
            pass
        else:
            self.expert_prompt_image_pool = checkpoints['expert_prompt_image_pool']
            self.expert_prompt_text_pool = checkpoints['expert_prompt_text_pool']

    def save(self, args, name='hash_model.pth'):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        """
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save({'model_state_dict': self.state_dict(),
                    'expert_prompt_image_pool': self.expert_prompt_image_pool,
                    'expert_prompt_text_pool': self.expert_prompt_text_pool
                    }, args.checkpoints_dir + name)
        return name

    def forward(self, *input):
        pass


class ImageModule(BasicModule):
    def __init__(self, args):
        super(ImageModule, self).__init__()
        self.attention = MultiheadAttention(embed_dim=512, num_heads=8)
        self.single_layer = nn.Linear(args.feature_dim, args.map_dim)
        self.features = nn.Sequential(
                # 1 full_conv1
                nn.Conv2d(in_channels=args.feature_dim+args.map_dim, out_channels=4096, kernel_size=1),
                # 2 relu1
                nn.ReLU(inplace=True),
                # 3 full_conv2
                nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
                # 4 relu2
                nn.ReLU(inplace=True),
            )
        # fc3
        self.classifier = nn.Linear(in_features=4096, out_features=args.bit)

    def forward(self, x, args):
        x_p = self.attention(x[:, args.feature_dim:2*args.feature_dim], x[:, 2*args.feature_dim:], x[:, 2*args.feature_dim:])
        x_p = self.single_layer(x_p[0])
        x = torch.cat((x[:, 0:args.feature_dim], x_p), dim=1)
        x = x.view(x.size(0), args.feature_dim+args.map_dim, 1, 1)
        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x)
        x = F.normalize(x)
        return x


class TextModule(BasicModule):
    def __init__(self, args):
        """
        :param y_dim: dimension of texts
        :param bit: bit number of the final binary code
        """
        super(TextModule, self).__init__()
        self.attention = MultiheadAttention(embed_dim=512, num_heads=8)
        self.single_layer = nn.Linear(args.feature_dim, args.map_dim)
        self.features = nn.Sequential(
                # 1 full_conv1
                nn.Conv2d(in_channels=args.feature_dim+args.map_dim, out_channels=4096, kernel_size=1),
                # 2 relu1
                nn.ReLU(inplace=True),
                # 3 full_conv2
                nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
                # 4 relu2
                nn.ReLU(inplace=True),
            )
        # fc3
        self.classifier = nn.Linear(in_features=4096, out_features=args.bit)

    def forward(self, x, args):
        x_p = self.attention(x[:, args.feature_dim:2*args.feature_dim], x[:, 2*args.feature_dim:], x[:, 2*args.feature_dim:])
        x_p = self.single_layer(x_p[0])
        x = torch.cat((x[:, 0:args.feature_dim], x_p), dim=1)
        x = x.view(x.size(0), args.feature_dim+args.map_dim, 1, 1)

        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x)
        x = F.normalize(x)
        return x
    

class MyModel(BasicModule):
    def __init__(self, args, task_index):
        super(MyModel, self).__init__()
        torch.manual_seed(args.seed)
        
        # 初始化全局提示
        self.global_prompt_image = nn.Parameter(torch.randn(1, args.feature_dim)).cuda()
        self.global_prompt_text = nn.Parameter(torch.randn(1, args.feature_dim)).cuda()
        # 将全局提示（非叶子张量）转换为叶子张量
        self.global_prompt_image = nn.Parameter(self.global_prompt_image.clone().detach().requires_grad_(True))
        self.global_prompt_text = nn.Parameter(self.global_prompt_text.clone().detach().requires_grad_(True))
        
        # 专家提示
        self.expert_prompt_text_pool = {}
        self.expert_prompt_image_pool = {}

        for i in range(task_index):
            # 为每个任务初始化空列表
            self.expert_prompt_image_pool[i] = []
            self.expert_prompt_text_pool[i] = []
            # 为加载的模型添加当前任务的专家提示
            for _ in range(args.num_prompt):
                self.expert_prompt_image_pool[i].append(nn.Parameter(torch.randn(1, args.feature_dim).clone().detach().requires_grad_(True)).cuda())
                self.expert_prompt_text_pool[i].append(nn.Parameter(torch.randn(1, args.feature_dim).clone().detach().requires_grad_(True)).cuda())

        self.image_net = ImageModule(args)
        self.text_net = TextModule(args)
    
    def forward(self, args, image_feature, text_feature, task_index, separation):
        if self.training == True:
            image_prompt_feature = train_select_prompt(args, image_feature, self.expert_prompt_image_pool, task_index)
            text_prompt_feature = train_select_prompt(args, text_feature, self.expert_prompt_text_pool, task_index)
            global_prompt_image_repeated = self.global_prompt_image.repeat(image_feature.shape[0], 1).cuda()
            global_prompt_text_repeated = self.global_prompt_text.repeat(text_feature.shape[0], 1).cuda()
            image_prompt_feature = torch.cat((image_feature, global_prompt_image_repeated, image_prompt_feature), dim=1)
            text_prompt_feature = torch.cat((text_feature, global_prompt_text_repeated, text_prompt_feature), dim=1)
        elif self.training == False:
            if separation == False:
                image_prompt_feature = valid_select_prompt(args, image_feature, self.expert_prompt_image_pool, task_index).cuda()
                text_prompt_feature = valid_select_prompt(args, text_feature, self.expert_prompt_text_pool, task_index).cuda()
            else:
                image_prompt_feature = train_select_prompt(args, image_feature, self.expert_prompt_image_pool, task_index)
                text_prompt_feature = train_select_prompt(args, text_feature, self.expert_prompt_text_pool, task_index)
            global_prompt_image_repeated = self.global_prompt_image.repeat(image_feature.shape[0], 1).cuda()
            global_prompt_text_repeated = self.global_prompt_text.repeat(text_feature.shape[0], 1).cuda()
            image_prompt_feature = torch.cat((image_feature, global_prompt_image_repeated, image_prompt_feature), dim=1)
            text_prompt_feature = torch.cat((text_feature, global_prompt_text_repeated, text_prompt_feature), dim=1)
        image_hash = self.image_net(image_prompt_feature, args)
        text_hash = self.text_net(text_prompt_feature, args)
        return image_hash, text_hash