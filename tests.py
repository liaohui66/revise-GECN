# analyze_and_clean.py
# 终极整合脚本：自动分析、筛选并生成干净的数据集文件

import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import math

try:
    # 直接调用您项目中的真实数据加载函数
    from data_utils import create_pyg_data
    from pymatgen.core.structure import Structure
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保此脚本与 data_utils.py 在同一目录下。")
    exit(1)

def main():
    # --- 配置区 ---
    input_filename = "4201358.json"
    output_filename = "4201358_cleaned.json"
    
    # create_pyg_data 函数需要的参数
    RADIAL_CUTOFF = 5.0
    DEFAULT_DTYPE = torch.float32

    # --- 1. 加载原始数据 ---
    print(f"--- 步骤 1/3: 加载原始数据集 '{input_filename}' ---")
    file_path = Path(input_filename)
    if not file_path.exists(): 
        print(f"错误: 文件 '{file_path}' 未找到。")
        return
        
    print("正在加载JSON文件到内存...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            structures_list = full_data["structure"]
            piezo_list = full_data["total"]
        print(f"文件加载完毕，共 {len(structures_list)} 个样本。")
    except Exception as e:
        print(f"加载或解析JSON文件时出错: {e}")
        return
    
    # --- 2. 分析阶段：计算所有边数并确定阈值 ---
    print(f"\n--- 步骤 2/3: 分析边数分布以确定筛选阈值 ---")
    edge_counts_report = []
    pbar_analyze = tqdm(
        enumerate(structures_list), 
        total=len(structures_list), 
        desc="分析进度"
    )
    
    for i, struct_dict in pbar_analyze:
        try:
            pymatgen_structure = Structure.from_dict(struct_dict)
            data_obj = create_pyg_data(
                pymatgen_structure=pymatgen_structure,
                piezo_tensor_target=piezo_list[i],
                atom_features_lookup=np.array([]),
                atom_feature_dim=1,
                radial_cutoff=RADIAL_CUTOFF,
                dtype=DEFAULT_DTYPE
            )
            edge_counts_report.append({
                "index": i, "num_edges": data_obj.num_edges
            })
        except Exception:
            # 在分析阶段，暂时忽略无法处理的结构
            edge_counts_report.append({"index": i, "num_edges": -1}) # 标记为错误
    
    # 从有效结果中提取边数
    valid_edges = np.array([item['num_edges'] for item in edge_counts_report if item['num_edges'] != -1])
    
    # 自动决策阈值：取99.5%百分位数的值，并向上取整到最近的百位数，增加一点余量
    percentile_val = np.percentile(valid_edges, 95.0)
    # 使用 math.ceil 向上取整，然后除以100再乘以100
    auto_threshold = math.ceil(percentile_val / 100) * 100
    
    print(f"分析完成。95.0%的样本边数在 {percentile_val:.0f} 以下。")
    print(f"自动确定的筛选阈值为: {auto_threshold}")

    # --- 3. 清洗阶段：根据阈值筛选并生成新文件 ---
    print(f"\n--- 步骤 3/3: 根据阈值 {auto_threshold} 清洗数据集 ---")
    
    clean_structures = []
    clean_piezos = []
    blacklist_indices = []
    
    # 我们复用刚才计算好的报告，不再重新计算
    for report_item in tqdm(edge_counts_report, desc="筛选进度"):
        i = report_item["index"]
        num_edges = report_item["num_edges"]
        
        if num_edges != -1 and num_edges <= auto_threshold:
            # 这是一个好样本，保留它的原始数据
            clean_structures.append(structures_list[i])
            clean_piezos.append(piezo_list[i])
        else:
            # 这是一个坏样本（边数超标或处理失败），记录它的索引
            blacklist_indices.append(i)

    # --- 创建并保存新的干净JSON文件 ---
    cleaned_data_dict = {
        "structure": clean_structures,
        "total": clean_piezos
        # 如果原始文件还有其他键需要保留，可以在这里添加
    }

    print("\n" + "="*80)
    print("--- 清洗完成！最终报告 ---")
    print(f"原始样本数:     {len(structures_list)}")
    print(f"清洗后样本数:   {len(clean_structures)}")
    print(f"被剔除的样本数: {len(blacklist_indices)}")
    
    if blacklist_indices:
        print("\n被剔除的样本索引列表 (前100个):")
        print(sorted(blacklist_indices)[:100]) # 只打印前100个，避免刷屏

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data_dict, f, indent=2) # 使用 indent=2 增加可读性
        print(f"\n[SUCCESS] 已生成干净的数据集文件: '{output_filename}'")
        print("您现在可以在您的 train.py 中将文件名指向这个新文件进行训练。")
    except Exception as e:
        print(f"\n[ERROR] 保存新的JSON文件时失败: {e}")
    
    print("="*80)

if __name__ == "__main__":
    main()