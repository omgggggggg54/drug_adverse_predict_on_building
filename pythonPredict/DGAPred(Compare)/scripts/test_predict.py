import os
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
import pickle
from model import DGAPred
from sklearn import metrics
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# 导入main.py中的函数
from main import load_label, load_drug_feature, load_adr_feature, Extract_positive_negative_samples


def load_model(drug_feature, side_feature,args):
    """加载模型"""
    model_path = os.path.join(args.rawpath, "forTest", "CS_CGI_MESH_DGA9222", "model_fold3.pkl")
    print(f"尝试加载模型: {model_path}")
    
    # 创建模型实例，使用实际特征维度
    if drug_feature is not None and side_feature is not None:
        drug_dim = drug_feature[0].shape[0]*2
        side_dim = side_feature[0].shape[0]*2
        print(f"使用实际特征维度: drug_dim={drug_dim}, side_dim={side_dim}")
        model = DGAPred(drug_dim, side_dim, 128, 128).to(device)
   
    # 加载模型权重
    with open(model_path, 'rb') as f:
        obj = f.read()
    weights = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}
    model.load_state_dict(weights)
    
    return model

def evaluate_model(ground_truth, label_truth, pred1, pred2):
    """评估模型性能
    
    Args:
        ground_truth: 原始标签值（可能包含连续值）
        label_truth: 二分类标签（0或1）
        pred1: 二分类预测值，用于计算AUC和AUPR
        pred2: 回归预测值，用于计算RMSE和MAE
        
    Returns:
        i_auc: ROC曲线下面积
        iPR_auc: 精确率-召回率曲线下面积
        rmse: 均方根误差
        mae: 平均绝对误差
    """
    # 使用pred1(二分类输出)计算AUC和AUPR
    # 计算精确率-召回率曲线和AUC
    iprecision, irecall, ithresholds = metrics.precision_recall_curve(
        label_truth, pred1, pos_label=1, sample_weight=None
    )
    iPR_auc = metrics.auc(irecall, iprecision)
    
    # 计算ROC AUC
    try:
        i_auc = metrics.roc_auc_score(label_truth, pred1)
    except ValueError:
        i_auc = 0
    
    # 使用pred2(回归输出)计算RMSE和MAE
    # 只对真实标签为正的样本计算
    one_label_index = np.nonzero(label_truth)
    rmse = sqrt(mean_squared_error(pred2[one_label_index], ground_truth[one_label_index]))
    mae = mean_absolute_error(pred2[one_label_index], ground_truth[one_label_index])
    
    return i_auc, iPR_auc, rmse, mae

def split_train_test(drug_feature, side_feature,  data_test):
    """与data_utils.py中的实现保持一致"""

    data_test = np.array(data_test)

    # 水平拼接药物特征
    drug_features_matrix = drug_feature[0]
    for i in range(1, len(drug_feature)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_feature[i]))

    # 水平拼接副作用特征
    side_features_matrix = side_feature[0]
    for i in range(1, len(side_feature)):
        side_features_matrix = np.hstack((side_features_matrix, side_feature[i]))

    # 提取测试数据
    if len(data_test) > 0:
        drug_test = drug_features_matrix[data_test[:, 0]]
        side_test = side_features_matrix[data_test[:, 1]]
        f_test = data_test[:, 2]
    else:
        drug_test = []
        side_test = []
        f_test = []

    
    return drug_test, side_test, f_test

def test(model, test_loader, device):
    """测试函数，从main.py中提取并简化"""
    model.eval()
    pred1 = []
    pred2 = []
    ground_truth = []
    label_truth = []
    
    with torch.no_grad():
        for step, (test_drug, test_side, test_ratings) in enumerate(test_loader):
            test_labels = test_ratings.clone().long()
            for k in range(test_ratings.data.size()[0]):
                if test_ratings.data[k] > 0:
                    test_labels.data[k] = 1
            
            scores_one, scores_two = model(test_drug, test_side, device)
            
            pred1.append(list(scores_one.data.cpu().numpy()))
            pred2.append(list(scores_two.data.cpu().numpy()))
            ground_truth.append(list(test_ratings.data.cpu().numpy()))
            label_truth.append(list(test_labels.data.cpu().numpy()))

    pred1 = np.array(sum(pred1, []), dtype=np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)
    ground_truth = np.array(sum(ground_truth, []), dtype=np.float32)
    label_truth = np.array(sum(label_truth, []), dtype=np.float32)

    return ground_truth, pred1, pred2, label_truth

def predict_with_original_data(model, test_loader, device, args):
   
    # 进行预测
    print("开始预测...")
    ground_truth, pred1, pred2, label_truth = test(model, test_loader, device)
    
    # 评估结果
    i_auc, iPR_auc, rmse, mae = evaluate_model(ground_truth, label_truth, pred1, pred2)
    
    # 输出结果
    print("\n========== 模型性能评估结果 ==========")
    print(f"使用pred1(二分类输出)计算:")
    print(f"AUC: {i_auc:.5f}")
    print(f"AUPR: {iPR_auc:.5f}")
    print(f"使用pred2(回归输出)计算:")
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE: {mae:.5f}")
    print("====================================")
    
    

if __name__ == "__main__":
    # 设置GPU
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print(f"使用设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("未使用GPU")

    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("开始模型性能测试...")
    parser = argparse.ArgumentParser(description="模型测试")
    parser.add_argument("--rawpath", type=str, default=current_dir+"/data/")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    print(f"当前工作目录: {args.rawpath}")
    print("开始加载原始数据...")
    
    # 加载药物列表
    druglist = pd.read_csv(os.path.join(args.rawpath, "lincs_druglist_ge_go_521.csv"))
    
    # 加载标签数据
    remain_drug_list, adr_list, drug_side = load_label(druglist["pert_id"], True, True, args)
    print(f"药物-副作用矩阵形状: {pd.DataFrame(drug_side).shape}")
    
    # 提取正负样本
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side.values, addition_negative_number='all')
    final_sample = np.vstack((final_positive_sample, final_negative_sample))
    
    # 准备数据
    data = []
    for i in range(final_sample.shape[0]):
        data.append((final_sample[i, 0], final_sample[i, 1], final_sample[i, 2]))
    
    # 加载特征
    print("加载药物和副作用特征...")
    drug_feature = load_drug_feature(remain_drug_list, args)
    side_feature = load_adr_feature(adr_list, args)
    
    # 加载模型
    print("加载模型...")
    model = load_model(drug_feature, side_feature,args)
    
    # 准备测试数据
    drug_test, side_test, f_test = split_train_test(drug_feature, side_feature, data)
    
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), 
                                            torch.FloatTensor(side_test),
                                            torch.FloatTensor(f_test))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=4, pin_memory=True)
    
    start_time = time.time()
    

    predict_with_original_data(model, test_loader, device, args)
    
    print(f"测试用时: {time.time() - start_time:.2f}秒")
