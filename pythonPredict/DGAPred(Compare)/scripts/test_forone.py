# 导入所需的库
import os
import sys
import os.path as osp
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")
import time
# 用于处理数据的工具
import random
import pandas as pd 
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error # 评分 MAE 的计算函数
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split # 拆分训练集与验证集工具
from data_utils import *
from model import DGAPred
from clac_dis_mesh_sim import cal_SimilarityByMeSHDAG
import pickle
#from alive_progress import alive_bar # 显示循环的进度条工具
# import sweetviz as sv
import warnings

def load_label(screen_drug_list,use_DGen,use_AGen,args):
    pd_label=pd.read_csv(args.rawpath+"sider_pert_mesh_list.csv",header=0,delimiter='\t') # 原始训练数据
    drug_col="pert_id"
    adr_col="MESH_ID"
    drug_side=data_feature(pd_label,screen_list=screen_drug_list,screen_col=drug_col,del_screen_col=False)
    #筛选有CTD GENE特征的Drug
    if use_DGen:
        pd_DGen=pd.read_csv(args.rawpath+"ctd_chem_pert_gene_ixns_list.csv",header=0,delimiter='\t')
        drug_list_DGen=sorted(np.unique(pd_DGen.loc[:,drug_col].values).tolist())
        drug_side=pd.concat([drug_side.loc[drug_side[drug_col]==id] for id in drug_list_DGen],ignore_index=True)
    drug_list=sorted(np.unique(drug_side.loc[:,"pert_id"].values).tolist())
    #筛选有CTD GENE特征的ADR
    if use_AGen:
        pd_AGen=pd.read_csv(args.rawpath+"ctd_gene_adr_asso_list_4386.csv",header=0,delimiter='\t')
        adr_list_Gen=sorted(np.unique(pd_AGen.loc[:,adr_col].values).tolist())
        drug_side=pd.concat([drug_side.loc[drug_side[adr_col]==id] for id in adr_list_Gen],ignore_index=True)

    drug_side=Convert_triplelist2matrix(drug_side,["pert_id","MESH_ID","label"],fillna_val=0) 
    adr_list = list(drug_side.columns)
    return drug_list,adr_list,drug_side

def load_drug_feature(screen_drug_list,args):
    drug1_smiles='CCC(C)C(C(=O)NC(C)C(=O)NC(CC1=CNC2=CC=CC=C21)C(=O)NC(CC(C)C)C(=O)NC(C(C)C)C(=O)NC(CCCCN)C(=O)NCC(=O)NC(CCCNC(=N)N)C(=O)N)NC(=O)C(CC3=CC=CC=C3)NC(=O)C(CCC(=O)O)NC(=O)C(CCCCN)NC(=O)C(C)NC(=O)C(C)NC(=O)C(CCC(=O)N)NC(=O)CNC(=O)C(CCC(=O)O)NC(=O)C(CC(C)C)NC(=O)C(CC4=CC=C(C=C4)O)NC(=O)C(CO)NC(=O)C(CO)NC(=O)C(C(C)C)NC(=O)C(CC(=O)O)NC(=O)C(CO)NC(=O)C(C(C)O)NC(=O)C(CC5=CC=CC=C5)NC(=O)C(C(C)O)NC(=O)CNC(=O)C(CCC(=O)O)NC(=O)C(C)NC(=O)C(CC6=CN=CN6)N'
    drug_features = []
    #CS Feature
    pd_cs=pd.read_csv(args.rawpath+"drug_pert_similes_list.csv",header=0,delimiter='\t')
    drug_col="pert_id"
    drug_smiles=data_feature(pd_cs,screen_list=screen_drug_list,screen_col=drug_col)
    drug_cs=get_SMILES_Similarity_forone(drug1_smiles,drug_smiles)
    print('drug CS shape:',drug_cs.shape)

    #CTD DGEN Feature
    pd_DGen=pd.read_csv(args.rawpath+"ctd_chem_pert_gene_ixns_list.csv",header=0,delimiter='\t')
    drug_col="pert_id"
    drug_DGen=data_feature(pd_DGen,screen_list=screen_drug_list,screen_col=drug_col,del_screen_col=False)
    # drug_DGen=pd.concat([drug_DGen['pert_id'],drug_DGen['GeneSymbol']]).unique()    #去除Action列，统一为1：存在相互作用
    drug_DGen=drug_DGen.drop_duplicates(subset=['pert_id','GeneSymbol'],keep='first')    #去除Action列，统一为1：存在相互作用
    #drug_DGen=pd.DataFrame(drug_DGen).insert(loc=2,column="ixn",value=1)                 #1:药物-基因存在相互作用
    drug_DGen["ixn"]=1
    drug_DGen=Convert_triplelist2matrix(drug_DGen,["pert_id","GeneSymbol","ixn"],fillna_val=0)#模型的药物-基因相互作用矩阵
    # pd.DataFrame(drug_DGen).to_csv(args.rawpath+"forTest/mat_drug_DGen.csv",header=True,index=True)
    # 增加测试的药物-基因相互关系 begin*********************************************************
    pd_genelist = pd.read_csv(args.rawpath+"forTest/CTD_D052216_genes_20240514103700.csv") 
    genelist=pd_genelist["Gene Symbol"].tolist()
    new_row=pd.Series(np.zeros(drug_DGen.shape[1]),index=drug_DGen.columns)
    new_row.loc[genelist]=1
    drug_DGen=drug_DGen.append(new_row,ignore_index=True)
    # 增加测试的药物-基因相互关系 end*********************************************************
    drug_DGen_sim = jaccard_similarity(drug_DGen)

    # 只取测试的药物的相似性向量 begin*********************************************************
    drug_DGen_sim_one=drug_DGen_sim[drug_DGen_sim.shape[0]-1,0:drug_DGen_sim.shape[0]-1]
    drug_DGen_sim_one=drug_DGen_sim_one.reshape(1,drug_DGen_sim.shape[0]-1)
    # 只取测试的药物的相似性向量 begin*********************************************************
    # pd.DataFrame(drug_DGen_sim).to_csv("mat_drug_DGen.csv",header=True,index=True)
    print('drug DGen shape:',drug_DGen_sim_one.shape)

    #LINCS GE Feature
    pd_ge=pd.read_csv(args.rawpath+"LINCS_Gene_Experssion_signatures_CD.csv",header=0,delimiter=',')
    drug_col="pert_id"
    drug_ge=data_feature(pd_ge,screen_list=screen_drug_list,screen_col=drug_col)
    drug_ge_sim = cosine_similarity(drug_ge)
    print('drug GE shape:',drug_ge_sim.shape)#表达量
    

    drug_features.append(drug_cs)
    drug_features.append(drug_DGen_sim_one)
    # drug_features.append(drug_ge_sim)
    return drug_features

def load_adr_feature(screen_adr_list,args):
    side_features = []
    adr_list=pd.DataFrame(screen_adr_list,columns=["MESH_ID"])  #格式：MESH:D002311
    adr_list_id=adr_list["MESH_ID"].str.replace("MESH:","")     #格式：D002311
    #MESH feature
    pd_label=pd.read_csv(args.rawpath+"se_mesh_dict_list.csv",header=0,delimiter='\t') # 原始训练数据
    adr_col="MESH_ID"
    side_mesh=data_feature(pd_label,screen_list=adr_list_id.values,screen_col=adr_col,del_screen_col=False)
    # pd.DataFrame(side_mesh).to_csv(args.rawpath+"forTest/data_adr_mesh.csv",header=True,index=True)
    side_mesh_sim=get_MESH_Similarity(side_mesh)

    #CTD Gene-Disease Feature
    pd_GDisease=pd.read_csv(args.rawpath+"ctd_gene_adr_asso_list_4386.csv",header=0,delimiter='\t')    #原数据已去重
    adr_GDisease=data_feature(pd_GDisease,screen_list=adr_list["MESH_ID"],screen_col=adr_col,del_screen_col=False)
    adr_GDisease=Convert_triplelist2matrix(adr_GDisease,["MESH_ID","GeneSymbol","ixn"],fillna_val=0)
    print(pd.DataFrame(adr_GDisease).shape)
    # pd.DataFrame(adr_GDisease).to_csv(args.rawpath+"forTest/mat_adr_Gen.csv",header=True,index=True)
    adr_GDisease_sim = jaccard_similarity(adr_GDisease)
    
    side_features.append(side_mesh_sim)
    side_features.append(adr_GDisease_sim)
    return side_features

def Extract_positive_negative_samples(DAL, addition_negative_number='all'):
    k = 0
    interaction_target = np.zeros((DAL.shape[0]*DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    number_negative = interaction_target.shape[0] - number_positive
    final_positive_sample = data_shuffle[number_negative::]
    negative_sample = data_shuffle[0:number_negative]
    a = np.arange(number_negative)#number_negative
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (number_negative)) #随机抽样N次/打乱顺序
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]##取和正样本一样多的负样本0~n
    addition_negative_sample = negative_sample[b[number_positive::], :]#除final_negative_sample之外的负样本
    return addition_negative_sample, final_positive_sample, final_negtive_sample


def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    '''
    稀疏多标签交叉熵损失的torch实现
    '''
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[...,:1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def fold_files(args):
    rawdata_dir = args.rawpath
    # 数据准备
    druglist = pd.read_csv(args.rawpath+"lincs_druglist_ge_go_521.csv") # 筛选实验药物
    
    remain_drug_list,adr_list, drug_side = load_label(druglist["pert_id"],True,True, args)
    print("drug_side shape:",pd.DataFrame(drug_side).shape)
    drug_side_sim = jaccard_similarity(drug_side.values)
    side_drug_sim = jaccard_similarity(drug_side.values.T)
    print("drug_side_sim : ",drug_side_sim.shape)
    print("side_drug_sim : ",side_drug_sim.shape)
    drug_features=load_drug_feature(remain_drug_list, args)
    side_features=load_adr_feature(adr_list, args)

    
    # with open(args.rawpath+'forTest/Test_DrugList.txt', 'w') as file:
    #     for index, item in enumerate(remain_drug_list):
    #         file.write(f'{index}\t{item}\n')
    # with open(args.rawpath+'forTest/Test_ADRList.txt', 'w') as file:
    #     for index, item in enumerate(adr_list):
    #         file.write(f'{index}\t{item}\n')
    
     #增加Drug-Side相似性
    if args.use_drugside:
        # drug_side_sim = jaccard_similarity(drug_side.values)
        # side_drug_sim = jaccard_similarity(drug_side.values.T)
        drug_features.append(drug_side_sim)
        side_features.append(side_drug_sim)
    # drug_features, side_features = read_raw_data(rawdata_dir)

    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))

    two_cell = []
    for i in range(1):#drug_features_matrix.shape[0]#（只预测一个药物）
        for j in range(side_features_matrix.shape[0]):
            two_cell.append([i, j])

    # two_cell = two_cell[0:1000]

    two_cell = np.array(two_cell)

    drug_test = drug_features_matrix[two_cell[:, 0]]
    side_test = side_features_matrix[two_cell[:, 1]]

    return drug_test, side_test, two_cell,remain_drug_list,adr_list

def test_data(args):
    drug_test, side_test, two_cell,drug_list,adr_list = fold_files(args)
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test))
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    torch.backends.cudnn.benchmark = True
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = DGAPred(453*2, 1019*2, args.embed_dim, args.batch_size).to(device)
    weights_path = args.rawpath+"forTest/CS_CGI_MESH_DGA9222/model_fold3.pkl"
    with open(weights_path, 'rb') as f:
        obj = f.read()
    weights = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}
    model.load_state_dict(weights)
    # model = ConvNCF(7570, 3976, args.embed_dim, args.batch_size).to(device)
    # model_file = args.rawpath + '/' + 'my_model.dat'
    # cheeckpoint = torch.load(model_file, map_location = device)
    # model.load_state_dict(cheeckpoint['model'])
    model.eval()
    pred1 = []
    pred2 = []
    for test_drug, test_side in _test:
        scores_one, scores_two = model(test_drug, test_side, device)
        pred1.append(list(scores_one.data.cpu().numpy()))
        pred2.append(list(scores_two.data.cpu().numpy()))
    pred1 = np.array(sum(pred1, []), dtype=np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)

    print('Output_data')
    output = []
    output.append(['drug_id', 'side_effect_id', 'Sample_association_score'])#, 'Sample_frequency_score'
    for i in range(pred1.shape[0]):
        if pred1[i] < 0.5:
            pred2[i] = 0
        output.append(["D052216", str(adr_list[two_cell[i][1]]), str(pred1[i])])  #str(drug_list[two_cell[i][0]]), str(pred2[i])

    t = ''

    with open(args.rawpath+'forTest/DGA_Prediction_D052216.txt', 'w') as q:
        for i in output:
            for e in range(len(output[0])):
                t = t + str(i[e]) + '\t'
            q.write(t.strip(' '))
            q.write('\n')
            t = ''



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default = 200,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0005,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.00001,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--N', type = int, default = 30000,
                        metavar = 'N', help = 'L0 parameter')
    parser.add_argument('--droprate', type = float, default = 0.5,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--use_drugside', type = bool, default = False,
                        help = 'add drug-side similarity')
    parser.add_argument('--rawpath', type=str, default='E:/workspace/python/DGAPred(Compare)/2drug-2side/DGAPred/data/',
                        metavar='STRING', help='rawpath')
    args = parser.parse_args()

    
    # print('Dataset: ' + args.dataset)
    print('-------------------- Hyperparams --------------------')
    print('N: ' + str(args.N))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    test_data(args)
    '''
    # 数据准备
    druglist = pd.read_csv(args.rawpath+"lincs_druglist_ge_go_521.csv") # 筛选实验药物
    
    remain_drug_list,adr_list, drug_side = load_label(druglist["pert_id"],True,True, args)
    print("drug_side shape:",pd.DataFrame(drug_side).shape)
    # pd.DataFrame(drug_side).to_csv("drug_side.csv",header=True,index=True)
    # drug_side_report=sv.analyze(drug_side)
    # drug_side_report.show_html(filepath=f"drug_side_report{str(time.strftime('%Y%m%d%H%M'))}.html",open_browser=False)
    # adr_list=pd.DataFrame(adr_list,columns=["MESH_ID"])

    #不参与训练的负样本，len(final_positive_sample)=len(final_negative_sample)
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side.values, addition_negative_number='all')
    final_sample=np.vstack((addition_negative_sample,final_negative_sample))#只测试负样本
    # final_sample = np.vstack((final_positive_sample, final_negative_sample))#均衡抽样
    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []
    data_neg_x = []
    data_neg_y = []
    data_neg = []
    pred = []
    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1], addition_negative_sample[i, 2]))
    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 0], X[i, 1], X[i, 2]))
    drug_feature=load_drug_feature(remain_drug_list, args)
    side_feature=load_adr_feature(adr_list, args)
    
    # pd.DataFrame(data,columns=["drug","side","label"]).to_csv("Test_Label_List.csv",header=True,index=False)
    #测试############################################################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    #增加Drug-Side相似性
    if args.use_drugside:
        drug_side_sim = jaccard_similarity(drug_side.values)
        side_drug_sim = jaccard_similarity(drug_side.values.T)
        drug_feature.append(drug_side_sim)
        side_feature.append(side_drug_sim)

    model = DGAPred(drug_feature[0].shape[0]*3, side_feature[0].shape[0]*3, args.embed_dim, args.batch_size).to(device)
    weights_path = "E:/workspace/python/DGAPred_addSim(Compare)/results/CS_GEN_MESH_GEN/model_fold2.pkl"
    with open(weights_path, 'rb') as f:
        obj = f.read()
    weights = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}
    model.load_state_dict(weights)
    # model.load_state_dict(torch.load("E:/workspace/python/DGAPred_addSim(Compare)/results/CS_GEN_MESH_GEN/model_fold2.pkl"))
    ######################################
    drug_features_matrix = drug_feature[0]
    for i in range(1, len(drug_feature)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_feature[i]))
    side_features_matrix = side_feature[0]
    for i in range(1, len(side_feature)):
        side_features_matrix = np.hstack((side_features_matrix, side_feature[i]))
    # drug_test = drug_features_matrix[data[:, 0]]
    # side_test = side_features_matrix[data[:, 1]]
    # f_test = data[:, 2]
    ######################################
    test_drugs = []
    test_sides = []
    for _,(drug_index,side_index,label) in enumerate(data):
        # test_drug=torch.Tensor(drug_features_matrix[drug_index])
        # test_side=torch.Tensor(side_features_matrix[side_index])
        test_drug=torch.Tensor(drug_features_matrix[drug_index])
        test_side=torch.Tensor(side_features_matrix[side_index])
        test_drugs.append(test_drug)
        test_sides.append(test_side)
        if _ % 31 == 0:    
            test_drugs = torch.stack(test_drugs)
            test_sides = torch.stack(test_sides)
            print("test_drugs:{}".format(test_drugs.shape))
            print("test_sides:{}".format(test_sides.shape))
            scores_one, scores_two = model(test_drugs, test_sides, device)
            pred.append(drug_index,side_index,scores_one,scores_two)
            print("Drug-Side-label-score1-score2:",drug_index,side_index,label,scores_one,scores_two)
            test_drugs = []
            test_sides = []
    # np.savetxt("Pred_list.txt",pred,delimiter='\t')
    pd.DataFrame(pred,columns=["drug","side","score_1","score_2"]).to_csv("Test_Pred_List.csv",header=True,index=False)()
    # testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
    #                                          torch.FloatTensor(f_test))
    # _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
    #                                     num_workers=16, pin_memory=True)    
    # Regression_criterion = nn.MSELoss()
    # Classification_criterion = nn.MSELoss()
    # t_i_auc, t_iPR_auc, t_rmse, t_mae, t_ground_i, t_ground_u, t_ground_truth, t_pred1, t_pred2, test_iter_loss, test_step = test(model,
    #                                                                                                        _test,
    #                                                                                                        _test,
    #                                                                                                        device,
    #                                                                                                        lossfunction1=Classification_criterion,
    #                                                                                                        lossfunction2=Regression_criterion)
    #测试############################################################################
    '''