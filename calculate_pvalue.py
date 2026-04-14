"""
P-Value Calculator for Model Comparison
计算两个模型（基线模型 vs 改进模型）在5折交叉验证上的统计显著性差异

使用方法:
    python calculate_pvalue.py --baseline <baseline_results.txt> --improved <improved_results.txt>
    
输入文件格式 (results.txt):
    Fold 1: AUC: 0.93252, AUPR: 0.92960, ACC: 0.86065, MCC: 0.72237
    Fold 2: AUC: 0.93103, AUPR: 0.92947, ACC: 0.85862, MCC: 0.71954
    ...

输出:
    生成 pvalue_report.txt 包含统计检验结果
"""

import argparse
import re
import numpy as np
from datetime import datetime
from scipy.stats import wilcoxon, ttest_rel


def parse_results_file(filepath):
    """解析results.txt文件，提取每折的指标值
    
    Args:
        filepath: results.txt文件路径
        
    Returns:
        dict: 包含各指标列表的字典 {metric_name: [fold1, fold2, ...]}
    """
    metrics = {
        'AUC': [],
        'AUPR': []
    }
    
    # 尝试解析ACC和MCC（改进模型有，原模型可能没有）
    has_acc_mcc = False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('Fold'):
            continue
            
        # 解析 "Fold X: AUC: 0.xxxxx, AUPR: 0.xxxxx, ACC: 0.xxxxx, MCC: 0.xxxxx"
        # 或 "Fold X: AUC: 0.xxxxx, AUPR: 0.xxxxx"
        
        # 提取AUC
        auc_match = re.search(r'AUC:\s*([\d.]+)', line)
        if auc_match:
            metrics['AUC'].append(float(auc_match.group(1)))
        
        # 提取AUPR
        aupr_match = re.search(r'AUPR:\s*([\d.]+)', line)
        if aupr_match:
            metrics['AUPR'].append(float(aupr_match.group(1)))
        
        # 提取ACC (如果有)
        acc_match = re.search(r'ACC:\s*([\d.]+)', line)
        if acc_match:
            if 'ACC' not in metrics:
                metrics['ACC'] = []
            metrics['ACC'].append(float(acc_match.group(1)))
            has_acc_mcc = True
        
        # 提取MCC (如果有)
        mcc_match = re.search(r'MCC:\s*([\d.]+)', line)
        if mcc_match:
            if 'MCC' not in metrics:
                metrics['MCC'] = []
            metrics['MCC'].append(float(mcc_match.group(1)))
    
    return metrics


def calculate_pvalues(baseline_metrics, improved_metrics):
    """计算两组指标的p-value
    
    使用两种统计检验方法:
    1. Wilcoxon符号秩检验 (非参数方法，推荐用于小样本)
    2. 配对t检验 (参数方法，假设数据服从正态分布)
    
    Args:
        baseline_metrics: 基线模型指标字典
        improved_metrics: 改进模型指标字典
        
    Returns:
        dict: 每个指标的p-value结果
    """
    results = {}
    
    # 获取两个模型共有的指标
    common_metrics = set(baseline_metrics.keys()) & set(improved_metrics.keys())
    
    for metric in common_metrics:
        baseline = np.array(baseline_metrics[metric])
        improved = np.array(improved_metrics[metric])
        
        if len(baseline) != len(improved):
            print(f"警告: {metric} 指标的折数不匹配 (baseline: {len(baseline)}, improved: {len(improved)})")
            continue
        
        if len(baseline) < 2:
            print(f"警告: {metric} 指标数据点不足，无法进行统计检验")
            continue
        
        # 计算差值
        diff = improved - baseline
        mean_diff = np.mean(diff)
        
        # Wilcoxon符号秩检验 (单侧: improved > baseline)
        try:
            # alternative='greater' 检验 improved > baseline
            wilcoxon_stat, wilcoxon_p = wilcoxon(improved, baseline, alternative='greater')
        except ValueError as e:
            # 如果所有差值为0，wilcoxon会报错
            wilcoxon_stat, wilcoxon_p = None, None
            print(f"Wilcoxon检验失败 ({metric}): {e}")
        
        # 配对t检验 (单侧)
        try:
            ttest_stat, ttest_p_twosided = ttest_rel(improved, baseline)
            # 单侧p值 (检验 improved > baseline)
            if mean_diff > 0:
                ttest_p = ttest_p_twosided / 2
            else:
                ttest_p = 1 - ttest_p_twosided / 2
        except Exception as e:
            ttest_stat, ttest_p = None, None
            print(f"t检验失败 ({metric}): {e}")
        
        results[metric] = {
            'baseline_mean': np.mean(baseline),
            'baseline_std': np.std(baseline),
            'improved_mean': np.mean(improved),
            'improved_std': np.std(improved),
            'mean_improvement': mean_diff,
            'wilcoxon_stat': wilcoxon_stat,
            'wilcoxon_p': wilcoxon_p,
            'ttest_stat': ttest_stat,
            'ttest_p': ttest_p,
            'baseline_values': baseline.tolist(),
            'improved_values': improved.tolist()
        }
    
    return results


def generate_report(results, baseline_path, improved_path, output_path):
    """生成p-value统计报告
    
    Args:
        results: 统计检验结果
        baseline_path: 基线模型结果文件路径
        improved_path: 改进模型结果文件路径  
        output_path: 输出报告路径
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        "=" * 80,
        "P-Value Statistical Comparison Report",
        "模型性能统计显著性检验报告",
        "=" * 80,
        f"生成时间: {timestamp}",
        f"基线模型结果: {baseline_path}",
        f"改进模型结果: {improved_path}",
        "",
        "统计检验方法:",
        "  1. Wilcoxon符号秩检验 (非参数方法，推荐用于小样本)",
        "  2. 配对t检验 (参数方法)",
        "  * p < 0.05 表示差异显著",
        "  * p < 0.01 表示差异非常显著",
        "",
        "=" * 80,
        "详细结果",
        "=" * 80,
        ""
    ]
    
    significance_summary = []
    
    for metric, data in results.items():
        report_lines.append(f"【{metric}】")
        report_lines.append("-" * 40)
        report_lines.append(f"  基线模型:   {data['baseline_mean']:.5f} ± {data['baseline_std']:.5f}")
        report_lines.append(f"  改进模型:   {data['improved_mean']:.5f} ± {data['improved_std']:.5f}")
        report_lines.append(f"  平均提升:   {data['mean_improvement']:+.5f} ({data['mean_improvement']/data['baseline_mean']*100:+.2f}%)")
        report_lines.append("")
        report_lines.append(f"  每折详情:")
        report_lines.append(f"    基线: {data['baseline_values']}")
        report_lines.append(f"    改进: {data['improved_values']}")
        report_lines.append("")
        
        # Wilcoxon结果
        if data['wilcoxon_p'] is not None:
            wilcoxon_sig = "***" if data['wilcoxon_p'] < 0.01 else ("**" if data['wilcoxon_p'] < 0.05 else "")
            report_lines.append(f"  Wilcoxon检验: p = {data['wilcoxon_p']:.6f} {wilcoxon_sig}")
        else:
            report_lines.append(f"  Wilcoxon检验: N/A")
        
        # t检验结果
        if data['ttest_p'] is not None:
            ttest_sig = "***" if data['ttest_p'] < 0.01 else ("**" if data['ttest_p'] < 0.05 else "")
            report_lines.append(f"  配对t检验:    p = {data['ttest_p']:.6f} {ttest_sig}")
        else:
            report_lines.append(f"  配对t检验:    N/A")
        
        report_lines.append("")
        
        # 显著性判断
        is_significant = False
        if data['wilcoxon_p'] is not None and data['wilcoxon_p'] < 0.05:
            is_significant = True
        if data['ttest_p'] is not None and data['ttest_p'] < 0.05:
            is_significant = True
        
        significance_summary.append({
            'metric': metric,
            'improvement': data['mean_improvement'],
            'wilcoxon_p': data['wilcoxon_p'],
            'ttest_p': data['ttest_p'],
            'is_significant': is_significant
        })
    
    # 总结
    report_lines.append("=" * 80)
    report_lines.append("总结 Summary")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("| 指标 | 提升 | Wilcoxon p | t-test p | 显著性 |")
    report_lines.append("|------|------|------------|----------|--------|")
    
    for item in significance_summary:
        wilcoxon_str = f"{item['wilcoxon_p']:.4f}" if item['wilcoxon_p'] is not None else "N/A"
        ttest_str = f"{item['ttest_p']:.4f}" if item['ttest_p'] is not None else "N/A"
        sig_str = "显著 ✓" if item['is_significant'] else "不显著"
        report_lines.append(f"| {item['metric']:4s} | {item['improvement']:+.4f} | {wilcoxon_str:10s} | {ttest_str:8s} | {sig_str} |")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("注: ** p<0.05, *** p<0.01")
    report_lines.append("=" * 80)
    
    # 写入文件
    report_content = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 同时打印到控制台
    print(report_content)
    
    return report_content


def main():
    parser = argparse.ArgumentParser(description='计算两个模型的p-value统计显著性')
    parser.add_argument('--baseline', type=str, required=True,
                        help='基线模型的results.txt路径')
    parser.add_argument('--improved', type=str, required=True,
                        help='改进模型的results.txt路径')
    parser.add_argument('--output', type=str, default='pvalue_report.txt',
                        help='输出报告路径 (默认: pvalue_report.txt)')
    
    args = parser.parse_args()
    
    print(f"解析基线模型结果: {args.baseline}")
    baseline_metrics = parse_results_file(args.baseline)
    print(f"  找到 {len(baseline_metrics['AUC'])} 折数据")
    
    print(f"解析改进模型结果: {args.improved}")
    improved_metrics = parse_results_file(args.improved)
    print(f"  找到 {len(improved_metrics['AUC'])} 折数据")
    
    print("\n计算p-value...")
    results = calculate_pvalues(baseline_metrics, improved_metrics)
    
    print(f"\n生成报告: {args.output}")
    generate_report(results, args.baseline, args.improved, args.output)
    
    print(f"\n报告已保存到: {args.output}")


if __name__ == '__main__':
    main()
