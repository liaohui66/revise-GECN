# debug_utils.py

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.data import Data

class DummyEncoder(nn.Module):
    def __init__(self, gnn_config: dict):
        super().__init__()
        # 我们需要从 config 中知道输出的 Irreps
        self.output_irreps = o3.Irreps(gnn_config["irreps_node_output"])
        # 创建一个可学习的、小的、固定的输出张量，以便有梯度流过
        # 我们让它的值很小，比如通过 N(0, 0.01) 初始化
        self.dummy_output_prototype = nn.Parameter(
            torch.randn(1, self.output_irreps.dim) * 0.01
        )
        print("\n" + "="*50)
        print("WARNING: Using DUMMY ENCODER! Real encoder is bypassed.")
        print(f"It will output features of irreps='{self.output_irreps}' and small, fixed values.")
        print("="*50 + "\n")

    def forward(self, data: Data) -> tuple[torch.Tensor, o3.Irreps]:
        # 输入 data 完全被忽略
        num_nodes = data.num_nodes
        
        # 将可学习的原型扩展到所有节点
        # expand 不会复制数据，内存高效
        dummy_features = self.dummy_output_prototype.expand(num_nodes, -1)
        
        # 返回与原 Encoder 格式相同的输出
        return dummy_features, self.output_irreps
    
class DummyIterativeBlock(nn.Module):
    """
    一个“绝对安全”的假迭代块，用于调试 SE3CapsuleConvBlock。
    它接收主胶囊的输出，但忽略大部分计算，直接返回一个
    数值稳定、尺度很小、形状正确的 P, A, F 张量。
    """
    def __init__(self, iterative_block_config: dict):
        super().__init__()
        # 从配置中获取最终输出的 Irreps，以便知道特征维度
        final_gconv_config = iterative_block_config["gconv_layer_configs"][-1]
        self.output_feature_irreps = o3.Irreps(final_gconv_config["irreps_node_output"])
        
        # 从配置中获取最后一层路由的输出胶囊数量
        final_routing_config = iterative_block_config["routing_layer_configs"][-1]
        self.num_output_capsules = final_routing_config["num_out_capsules"]

        # --- 创建可学习的原型，用于生成安全的输出 ---
        # 我们让输出的姿态接近于单位元（李代数为零向量），激活值为 0.5 (sigmoid(0))
        # 所有值都乘以 0.01 来确保尺度极小
        
        # 输出的 Pose
        self.dummy_pose_prototype = nn.Parameter(
            torch.randn(1, self.num_output_capsules, 6) * 0.01
        )
        # 输出的 Activation (在送入 sigmoid 之前)
        self.dummy_activation_prototype = nn.Parameter(
            torch.zeros(1, self.num_output_capsules)
        )
        # 输出的 Feature
        self.dummy_feature_prototype = nn.Parameter(
            torch.randn(1, self.num_output_capsules, self.output_feature_irreps.dim) * 0.01
        )
        
        print("\n" + "="*60)
        print("!!! WARNING: Using DUMMY ITERATIVE BLOCK for debugging !!!")
        print("    The real SE3CapsuleConvBlock is BYPASSED.")
        print(f"    It will output clean, small-valued P, A, and F.")
        print("="*60 + "\n")

    def forward(self, 
                current_P_alg,         # (B, M_in, 6)
                current_A,             # (B, M_in)
                base_node_features,    # (N_sum, D_gnn)
                base_node_positions,   # (N_sum, 3)
                base_edge_index,       # (2, E_sum)
                base_batch_idx,        # (N_sum,)
                current_F_caps         # (B, M_in, D_feat_in)
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 这个 forward 方法忽略所有复杂的输入，只关心 batch size
        batch_size = current_P_alg.shape[0]

        # 扩展原型以匹配 batch size
        final_P = self.dummy_pose_prototype.expand(batch_size, -1, -1)
        
        # 对激活值应用 sigmoid，使其在 (0, 1) 范围内，模拟真实输出
        final_A = torch.sigmoid(self.dummy_activation_prototype.expand(batch_size, -1))
        
        final_F = self.dummy_feature_prototype.expand(batch_size, -1, -1)

        # 返回与真实模块格式完全相同的元组
        return final_P, final_A, final_F

class DummyGroupConvLayer(nn.Module):
    """
    一个“绝对安全”的假群卷积层。
    它忽略所有复杂的几何和图计算，直接返回一个
    可学习的、小尺度的、正确形状的特征张量。
    """
    def __init__(self, irreps_node_output: str, **kwargs):
        super().__init__()
        self.output_irreps = o3.Irreps(irreps_node_output)
        
        # 创建一个可学习的输出原型，确保梯度可以流动
        # 乘以 0.01 确保其数值非常小，避免自身成为 NaN 源
        self.dummy_feature_prototype = nn.Parameter(
            torch.randn(1, self.output_irreps.dim) * 0.01
        )
        
        print("\n" + "="*60)
        print("!!! WARNING: Using DUMMY GROUP CONV LAYER for debugging !!!")
        print("    The real SE3GroupConvLayer is BYPASSED.")
        print(f"    It will output clean, small-valued features of irreps='{self.output_irreps}'.")
        print("="*60 + "\n")
        
    def forward(self, 
                input_node_features, 
                node_positions, 
                edge_index, 
                guiding_poses_algebra, 
                batch_idx_nodes):
        # guiding_poses_algebra 的形状是 (B, M_out, 6)
        B, M_out, _ = guiding_poses_algebra.shape
        
        # 忽略所有复杂的输入，只根据 B 和 M_out 生成正确的形状。
        # 真实模块的输出是池化后的特征，形状为 (B * M_out, D_feat_out)
        # 我们的 dummy_feature_prototype 是 (1, D_feat_out)
        # 所以我们可以用 expand 扩展到目标形状
        output_features = self.dummy_feature_prototype.expand(B * M_out, -1)
        
        return output_features
    
class DummyCapsuleLayer(nn.Module):
    """
    一个“绝对安全”的假胶囊路由层。
    它简单地将输入“直通”或通过一个简单的线性变换，
    完全绕过复杂的迭代路由和几何运算。
    """
    def __init__(self, num_in_capsules: int, num_out_capsules: int, **kwargs):
        super().__init__()
        self.num_in_capsules = num_in_capsules
        self.num_out_capsules = num_out_capsules
        
        # 为了匹配输出维度，我们需要一个简单的、可学习的变换
        # 我们只变换平移部分，旋转部分保持为零，以确保绝对稳定
        if num_in_capsules != num_out_capsules:
            self.pose_proj = nn.Linear(num_in_capsules * 3, num_out_capsules * 3)
            self.act_proj = nn.Linear(num_in_capsules, num_out_capsules)
        else:
            self.pose_proj = nn.Identity()
            self.act_proj = nn.Identity()
            
        print("\n" + "="*60)
        print("!!! WARNING: Using DUMMY CAPSULE LAYER for debugging !!!")
        print("    The real SE3CapsuleLayer (dynamic routing) is BYPASSED.")
        print("="*60 + "\n")

    def forward(self, input_poses_algebra: torch.Tensor, input_activations: torch.Tensor):
        # input_poses_algebra: (B, M_in, 6)
        # input_activations: (B, M_in)
        B, M_in, _ = input_poses_algebra.shape
        
        # --- 姿态处理 ---
        # 只取输入的平移部分 (前3维)，并将其展平
        trans_part_flat = input_poses_algebra[..., :3].reshape(B, -1)
        # 通过线性层进行维度匹配
        new_trans_part = self.pose_proj(trans_part_flat).reshape(B, self.num_out_capsules, 3)
        # 旋转部分强制设为0，确保稳定
        new_rot_part = torch.zeros_like(new_trans_part)
        # 组合成新的姿态
        output_poses_algebra = torch.cat([new_trans_part, new_rot_part], dim=-1)

        # --- 激活值处理 ---
        # 简单地通过线性和 sigmoid
        output_activations = torch.sigmoid(self.act_proj(input_activations))
        
        return output_poses_algebra, output_activations