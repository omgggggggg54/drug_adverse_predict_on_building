import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from resnet import Residual

class DGAPred(nn.Module):
    def __init__(self, drugs_dim, sides_dim, embed_dim, bathsize, dropout1=0.8, dropout2=0.8, n_drug_chunks=4, n_side_chunks=3):
        super(DGAPred, self).__init__()

        self.drugs_dim = drugs_dim
        self.sides_dim = sides_dim
        self.batchsize = bathsize
        self.n_drug_chunks = n_drug_chunks
        self.n_side_chunks = n_side_chunks
        self.drug_dim = self.drugs_dim // n_drug_chunks
        self.side_dim = self.sides_dim // n_side_chunks
        # print(self.drug_dim)
        # print(self.side_dim)
        # self.side_dim = side_dim
        # self.drug_dim = drug_dim
        self.embed_dim = embed_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        # self.drugs_layer_vgg=self.vgg_fc_layer(self.drugs_dim, self.embed_dim)
        self.drugs_layer = nn.Linear(self.drugs_dim, self.embed_dim)
        self.drugs_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drugs_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        # self.sides_layer_vgg=self.vgg_fc_layer(self.drugs_dim, self.embed_dim)
        self.sides_layer = nn.Linear(self.sides_dim, self.embed_dim)
        self.sides_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.sides_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        # 动态创建drug特征层
        self.drug_layers = nn.ModuleList()
        self.drug_layers_1 = nn.ModuleList()
        self.drug_bns = nn.ModuleList()
        for _ in range(n_drug_chunks):
            self.drug_layers.append(nn.Linear(self.drug_dim, self.embed_dim))
            self.drug_layers_1.append(nn.Linear(self.embed_dim, self.embed_dim))
            self.drug_bns.append(nn.BatchNorm1d(self.embed_dim, momentum=0.5))

        # 动态创建side特征层
        self.side_layers = nn.ModuleList()
        self.side_layers_1 = nn.ModuleList()
        self.side_bns = nn.ModuleList()
        for _ in range(n_side_chunks):
            self.side_layers.append(nn.Linear(self.side_dim, self.embed_dim))
            self.side_layers_1.append(nn.Linear(self.embed_dim, self.embed_dim))
            self.side_bns.append(nn.BatchNorm1d(self.embed_dim, momentum=0.5))

        self.num_class = 2
        # cnn setting
        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        self.number_map = n_drug_chunks * n_side_chunks     #drug feature num * sider feature map
        # self.resnet_interaction=self.make_layer(Residual,self.channel_size,3,self.kernel_size,self.strides)
        self.cnn_interaction = nn.Sequential(
            
            # batch_size * 1 * 64 * 64
            nn.Conv2d(self.number_map, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 2 * 2
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 1 * 1
        )
        self.residual_block = nn.Conv2d(self.number_map, self.channel_size, np.power(self.kernel_size,6), stride=np.power(self.strides,6))
        self.total_layer = nn.Linear((self.channel_size * 4 + 2 * self.embed_dim), self.channel_size * 4)
        self.total_bn = nn.BatchNorm1d((self.channel_size * 4 + 2 * self.embed_dim), momentum=0.5)
        self.classifier = nn.Linear(self.channel_size * 4, 1)
        self.con_layer = nn.Linear(self.channel_size * 4, 1)
        self.classifier2 = nn.Sequential(
            nn.Linear(self.channel_size * 4, 1),
            nn.BatchNorm1d(1, momentum=0.5),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(self.channel_size * 4, 1)  # 回归输出
    
    #重复同一个残差块
    def make_layer(self, block, channels, num_blocks,k_size, stride):
        # strides = [stride] + [1] * (num_blocks - 1)
        inchannel=self.number_map   #初始化inchannel为number_map
        layers = []
        for i in range(num_blocks):
            layers.append(block(inchannel, channels, k_size, stride))
            inchannel = channels    #后续inchannel为outchannel
        return nn.Sequential(*layers)


    def forward(self, drug_features, side_features, device):

        # x_drugs=self.drugs_layer_vgg(drug_features.to(device))
        x_drugs = F.relu(self.drugs_bn(self.drugs_layer(drug_features.to(device))), inplace=True)
        x_drugs = F.dropout(x_drugs, training=self.training, p=self.dropout1)
        x_drugs = self.drugs_layer_1(x_drugs)

        # x_sides=self.sides_layer_vgg(side_features.to(device))
        x_sides = F.relu(self.sides_bn(self.sides_layer(side_features.to(device))), inplace=True)
        x_sides = F.dropout(x_sides, training=self.training, p=self.dropout1)
        x_sides = self.sides_layer_1(x_sides)


        drug_chunks = drug_features.chunk(self.n_drug_chunks, 1)
        side_chunks = side_features.chunk(self.n_side_chunks, 1)

        # 动态处理drug特征
        drugs = []
        for i, chunk in enumerate(drug_chunks):
            x = F.relu(self.drug_bns[i](self.drug_layers[i](chunk.to(device))), inplace=True)
            x = F.dropout(x, training=self.training, p=self.dropout1)
            x = self.drug_layers_1[i](x)
            drugs.append(x)

        # 动态处理side特征
        sides = []
        for i, chunk in enumerate(side_chunks):
            x = F.relu(self.side_bns[i](self.side_layers[i](chunk.to(device))), inplace=True)
            x = F.dropout(x, training=self.training, p=self.dropout1)
            x = self.side_layers_1[i](x)
            sides.append(x)

        maps = []
        for i in range(len(drugs)):
            for j in range(len(sides)):
                maps.append(torch.bmm(drugs[i].unsqueeze(2), sides[j].unsqueeze(1)))

        interaction_map = maps[0].view((-1, 1, self.embed_dim, self.embed_dim))

        for i in range(1, len(maps)):
            interaction = maps[i].view((-1, 1, self.embed_dim, self.embed_dim))
            interaction_map = torch.cat([interaction_map, interaction], dim=1)
        # interaction_map = interaction_map.view((-1, 1, self.embed_dim, self.embed_dim))
        feature_map = self.cnn_interaction(interaction_map)  # output: batch_size * 32 * 1 * 1
        # feature_map = self.resnet_interaction(interaction_map)  # output: batch_size * 32 * 2 * 2
        # print(feature_map.size())
        h = feature_map.view((-1, 32*4))

        total = torch.cat((x_drugs, h, x_sides), dim=1)
        # print(total.size())

        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout2)

        classification = self.classifier2(total)
        regression = self.regressor(total)

        return classification.squeeze(), regression.squeeze()
