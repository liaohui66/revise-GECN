# group_conv.py

import torch
import torch.nn as nn
from e3nn import o3
from typing import List, Optional, Tuple
from torch_geometric.nn import global_mean_pool
from e3nn.math import orthonormalize
import math


def matrix_to_angles_custom(R: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    一个稳健的、支持批处理的函数，用于从旋转矩阵计算欧拉角(alpha, beta, gamma)。
    特别处理了万向节死锁(gimbal lock)的情况。
    """
    # R 的形状是 (..., 3, 3)
    beta = torch.acos(torch.clamp(R[..., 2, 2], -1.0 + eps, 1.0 - eps))

    # 检查是否接近万向节死锁 (beta 接近 0 或 pi)
    is_gimbal_lock = (beta < eps) | (beta > math.pi - eps)
    
    # 非死锁情况 (正常计算)
    alpha_normal = torch.atan2(R[..., 1, 2], R[..., 0, 2])
    gamma_normal = torch.atan2(R[..., 2, 1], -R[..., 2, 0])

    # 死锁情况 (beta=0 或 pi)，此时 alpha 和 gamma 耦合，我们约定 gamma=0
    alpha_gimbal = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    gamma_gimbal = torch.zeros_like(alpha_gimbal)
    
    # 根据掩码选择对应的结果
    alpha = torch.where(is_gimbal_lock, alpha_gimbal, alpha_normal)
    gamma = torch.where(is_gimbal_lock, gamma_gimbal, gamma_normal)
    
    return alpha, beta, gamma


try:
    from .encoder import TFNInteractionBlock, RadialMLP
    from .geometry_utils import se3_exp_map_custom, se3_inverse_custom
except ImportError:
    from encoder import TFNInteractionBlock, RadialMLP
    from geometry_utils import se3_exp_map_custom, se3_inverse_custom

# SVD and other helpers are assumed to be correct and kept as is.
from torch.autograd import Function

class SVD_with_Stable_Grad(Function):
    @staticmethod
    def forward(ctx, A):

        if A.dtype == torch.float16: A = A.to(torch.float32)
        try: U, S, Vh = torch.linalg.svd(A); ctx.save_for_backward(U, Vh); return U, S, Vh
        except torch.linalg.LinAlgError:
            batch_size, _, _ = A.shape; eye = torch.eye(3, device=A.device, dtype=A.dtype)
            U = eye.unsqueeze(0).expand(batch_size, -1, -1); S = torch.ones(batch_size, 3, device=A.device, dtype=A.dtype); Vh = eye.unsqueeze(0).expand(batch_size, -1, -1)
            ctx.save_for_backward(U, Vh); return U, S, Vh
    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_Vh):
        U, Vh = ctx.saved_tensors; return U @ grad_Vh

class SVD_with_Identity_Grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R_matrix):
        """
        前向传播：执行 SVD 并“清洗”矩阵。
        """
        try:
            # 强制使用 float32 以提高 SVD 的稳定性
            U, S, Vh = torch.linalg.svd(R_matrix.to(torch.float32))
            
            # 修正行列式以处理反射 (镜像变换)
            det = torch.det(U @ Vh)
            # 注意: 这里的 inplace 修改是安全的，因为它在 forward 内部，
            # 且我们不会为 backward 保存 U
            U[det < 0] = -U[det < 0]
            
            # 重构出一个完美的、正交的旋转矩阵
            R_perfect = U @ Vh
            return R_perfect.to(R_matrix.dtype)
        except torch.linalg.LinAlgError:
            # 如果 SVD 失败 (例如输入是 NaN 或 Inf)，返回一个单位阵批次
            batch_size = R_matrix.shape[0]
            identity = torch.eye(3, device=R_matrix.device, dtype=R_matrix.dtype)
            return identity.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：直接返回输入的梯度。
        这相当于梯度流过了一个单位矩阵，梯度值被完整保留，但路径是通畅的。
        """
        return grad_output

class SE3GroupConvLayer(nn.Module):
    def __init__(self,
                 irreps_node_input: str,
                 irreps_node_output: str,
                 irreps_sh: str,
                 num_basis_radial: int,
                 radial_mlp_hidden_dims: List[int],
                 num_interaction_layers: int = 1,
                 # The 'interaction_block_type' and transformer args are now ignored
                 **kwargs 
                ):
        super().__init__()
        self.irreps_node_input_obj, self.irreps_node_output_obj, self.irreps_sh_obj = o3.Irreps(irreps_node_input), o3.Irreps(irreps_node_output), o3.Irreps(irreps_sh)
        self.num_basis_radial, self.num_interaction_layers, self.eps = num_basis_radial, num_interaction_layers, 1e-8
        self.spherical_harmonics = o3.SphericalHarmonics(self.irreps_sh_obj, normalize=True, normalization='component')
        self.radial_embedding = RadialMLP(1, radial_mlp_hidden_dims, self.num_basis_radial)
        self.interaction_layers_module = nn.ModuleList()
        current_block_input_irreps = self.irreps_node_input_obj
        for i in range(self.num_interaction_layers):
            block_output_irreps = current_block_input_irreps if i < self.num_interaction_layers - 1 else self.irreps_node_output_obj
            # --- ONLY MODIFICATION: Force TFN Block Creation ---
            block = TFNInteractionBlock(
                irreps_node_input=str(current_block_input_irreps),
                irreps_node_output=str(block_output_irreps),
                irreps_edge_attr="0e",
                irreps_sh=str(self.irreps_sh_obj),
                num_basis_radial=self.num_basis_radial,
                radial_mlp_hidden=radial_mlp_hidden_dims
            )
            self.interaction_layers_module.append(block)
            current_block_input_irreps = block_output_irreps


    _debug_print_counter = 0

    # def _get_D_from_matrix_stable(self, irreps_obj, R_matrix):
    #     """
    #     终极杀手锏版。通过 .detach() 主动切断不稳定的梯度来源，保证训练的绝对稳定。
    #     """
    #     device = R_matrix.device
    #     original_dtype = R_matrix.dtype
    #     batch_size = R_matrix.shape[0]

    #     # --- Debug Printing ---
    #     if SE3GroupConvLayer._debug_print_counter < 2:
    #         print("\n" + "="*80)
    #         print(f"DEBUG: Executing DETACHED `_get_D_from_matrix_stable` (Call #{SE3GroupConvLayer._debug_print_counter + 1})")
    #         print("       This confirms the NON-DIFFERENTIABLE PROJECTION logic is ACTIVE.")
    #         is_first_call = True
    #         SE3GroupConvLayer._debug_print_counter += 1
    #     else:
    #         is_first_call = False

    #     try:
    #         # 我们在这里创建一个计算块，其中的所有梯度都不会影响到 R_matrix 之前的计算
    #         with torch.no_grad():
    #             # Step 1: Orthonormalize (已验证在GPU上可靠)
    #             R_ortho_list = []
    #             for i in range(batch_size):
    #                 ortho_mat, _ = orthonormalize(R_matrix[i].to(torch.float32))
    #                 R_ortho_list.append(ortho_mat)
    #             R_ortho_batch = torch.stack(R_ortho_list, dim=0)

    #             # Step 2: 使用我们自己的、可靠的函数获取角度
    #             alpha, beta, gamma = matrix_to_angles_custom(R_ortho_batch)

    #             # Step 3: 在CPU上安全地调用e3nn函数
    #             D_matrix_cpu = irreps_obj.D_from_angles(alpha.cpu(), beta.cpu(), gamma.cpu())
                
    #             # Step 4: 将结果移回原始设备
    #             D_matrix = D_matrix_cpu.to(device)

    #         # 这里的 D_matrix 不包含任何到 R_matrix 的梯度路径。
    #         # 但是，R_matrix 本身仍然是计算图的一部分，梯度会在模型的其他地方流过。
    #         # 我们需要将 D_matrix 连接回计算图，同时允许梯度流过 D_matrix 本身（用于后续层）
    #         # R_matrix.sum() * 0 确保 R_matrix 保持在计算图中，而 .detach() 确保了我们使用的是清理后的值
    #         return D_matrix.detach() + (R_matrix.sum() * 0).view(-1, 1, 1)


    #     except Exception as e:
    #         identity_D = torch.eye(irreps_obj.dim, device=device, dtype=original_dtype).unsqueeze(0).expand(batch_size, -1, -1)
    #         if is_first_call: print(f"       DETACHED METHOD FAILED (THIS SHOULD NOT HAPPEN): {e}. Returning identity D-matrix.")
    #         return identity_D

    def _get_D_from_matrix_stable(self, irreps_obj, R_matrix):
        """
        终极杀手锏版。通过 .detach() 主动切断不稳定的梯度来源，保证训练的绝对稳定。
        """
        device = R_matrix.device
        original_dtype = R_matrix.dtype
        batch_size = R_matrix.shape[0]

        if SE3GroupConvLayer._debug_print_counter < 2:
            print(f"DEBUG: Executing DETACHED `_get_D_from_matrix_stable`...")
            SE3GroupConvLayer._debug_print_counter += 1

        with torch.no_grad():
            alpha, beta, gamma = matrix_to_angles_custom(R_matrix.to(torch.float32))
            D_matrix_cpu = irreps_obj.D_from_angles(alpha.cpu(), beta.cpu(), gamma.cpu())
            D_matrix = D_matrix_cpu.to(device, dtype=original_dtype)
            
        # 关键：我们返回一个不带梯度的 D 矩阵，但加上一个与 R_matrix 相关的零项，
        # 确保 R_matrix 本身不会从计算图中掉队（以防万一有其他路径需要它）。
        return D_matrix + (R_matrix.sum() * 0).view(-1, 1, 1)

    # def _get_D_from_matrix_differentiable(self, irreps_obj, R_matrix):
    #     """
    #     一个可微分的版本，它信任 e3nn 的内置函数，并为极端情况提供回退。
    #     """
    #     device = R_matrix.device
    #     original_dtype = R_matrix.dtype
    #     batch_size = R_matrix.shape[0]

    #     try:
    #         D_matrix = irreps_obj.D_from_matrix(R_matrix.to(torch.float32))
    #         return D_matrix.to(original_dtype)
        
    #     except torch.linalg.LinAlgError as e:
    #         # 这是一个安全网，以防万一在某些数据上 SVD 失败。
    #         print(f"WARNING: e3nn.D_from_matrix failed with LinAlgError: {e}. "
    #             f"Returning identity D-matrix for this batch.")
    #         # 返回一个单位阵。为了保持计算图连接，我们添加一个依赖于输入的零项。
    #         identity_D = torch.eye(irreps_obj.dim, device=device, dtype=original_dtype)
    #         identity_D = identity_D.unsqueeze(0).expand(batch_size, -1, -1)
    #         # 这种方式确保 R_matrix 留在计算图中，但返回的是一个安全的单位阵。
    #         return identity_D + (R_matrix.sum() * 0).view(-1, 1, 1)

    # def _get_D_from_matrix_final(self, irreps_obj, R_matrix):
    #     """
    #     最终版本：使用自定义 SVD 函数来保证稳定性和可微性。
    #     """
    #     # 1. 使用我们的自定义 SVD 函数“清洗”矩阵
    #     R_perfect = SVD_with_Identity_Grad.apply(R_matrix)
        
    #     # 2. 将完美的 R_perfect 送入 e3nn
    #     try:
    #         D_matrix = irreps_obj.D_from_matrix(R_perfect)
    #         return D_matrix
    #     except Exception as e:
    #         print(f"WARNING: e3nn.D_from_matrix failed: {e}. Returning identity.")
    #         batch_size = R_matrix.shape[0]
    #         identity_D = torch.eye(irreps_obj.dim, device=R_matrix.device, dtype=R_matrix.dtype)
    #         return identity_D.unsqueeze(0).expand(batch_size, -1, -1) + (R_matrix.sum()*0).view(-1,1,1)

    def forward(self, input_node_features, node_positions, edge_index, guiding_poses_algebra, batch_idx_nodes):
        
        # --- “安全气囊”开始 ---
        # 我们在这里创建一个局部的、不使用 AMP 的上下文。
        # 即使全局的 autocast 是启用的，这个 with 块内的所有计算也会被强制使用 float32。
        device_type = input_node_features.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # 1. 将所有可能受 AMP 影响的输入张量，显式地转换为 float32
            input_node_features_f32 = input_node_features.to(torch.float32)
            node_positions_f32 = node_positions.to(torch.float32)
            guiding_poses_algebra_f32 = guiding_poses_algebra.to(torch.float32)
            
            # 2. 现在，在这个 float32 的“安全区”内，执行你原有的全部 forward 逻辑
            #    只需将所有输入变量替换为我们新创建的 _f32 版本
            
            Dev = input_node_features_f32.device
            B, M_out, _ = guiding_poses_algebra_f32.shape
            N_sum, E_sum = input_node_features_f32.shape[0], edge_index.shape[1]

            # 1. Pose Transformation
            flat_guiding_poses_alg = torch.clamp(guiding_poses_algebra_f32.reshape(B * M_out, 6), -10.0, 10.0)
            try:
                Guiding_Poses_Mat_Inv_flat = se3_inverse_custom(se3_exp_map_custom(flat_guiding_poses_alg))
            except Exception:
                Guiding_Poses_Mat_Inv_flat = torch.eye(4, device=Dev).unsqueeze(0).expand(B * M_out, -1, -1)
            R_mat = Guiding_Poses_Mat_Inv_flat[:, :3, :3]

            # 2. Feature Alignment
            super_batch_input_node_features_no_align = input_node_features_f32.repeat(M_out, 1)
            current_features = super_batch_input_node_features_no_align
            if self.irreps_node_input_obj.lmax > 0:
                D_R_inv = self._get_D_from_matrix_stable(self.irreps_node_input_obj, R_mat)
                original_graph_indices_expanded = batch_idx_nodes.repeat(M_out)
                view_indices_expanded = torch.arange(M_out, device=Dev).repeat_interleave(N_sum)
                pose_indices_for_nodes = original_graph_indices_expanded * M_out + view_indices_expanded
                D_R_inv_expanded = D_R_inv[pose_indices_for_nodes]
                # torch.bmm 现在在 float32 下运行，梯度将是稳定的
                aligned_features = torch.bmm(D_R_inv_expanded, super_batch_input_node_features_no_align.unsqueeze(-1)).squeeze(-1)
                current_features = torch.where(torch.isnan(aligned_features), super_batch_input_node_features_no_align, aligned_features)

            # 3. GNN Message Passing Preparation
            super_batch_pos = node_positions_f32.repeat(M_out, 1)
            edge_index_offsets = torch.arange(M_out, device=Dev) * N_sum
            super_batch_edge_index = edge_index.repeat(1, M_out) + edge_index_offsets.repeat_interleave(E_sum)
            row, col = super_batch_edge_index
            edge_vec = super_batch_pos[row] - super_batch_pos[col]
            edge_len = torch.norm(edge_vec, dim=1, keepdim=True)
            
            edge_sh_super_batch = torch.zeros(edge_vec.shape[0], self.irreps_sh_obj.dim, device=edge_vec.device, dtype=edge_vec.dtype)
            valid_edges_mask = (edge_len > self.eps).squeeze()
            if valid_edges_mask.any():
                edge_sh_super_batch[valid_edges_mask] = self.spherical_harmonics(edge_vec[valid_edges_mask] / edge_len[valid_edges_mask])
            
            edge_radial_emb_super_batch = self.radial_embedding(edge_len)

            # 4. TFN Interaction Loop
            for block in self.interaction_layers_module:
                current_features = block(
                    node_features=current_features,
                    edge_index=super_batch_edge_index,
                    edge_sh=edge_sh_super_batch,
                    edge_radial_emb=edge_radial_emb_super_batch
                )

            # 5. Pooling and Final Rotation
            pool_batch_idx = batch_idx_nodes.repeat(M_out) * M_out + torch.arange(M_out, device=Dev).repeat_interleave(N_sum)
            pooled_features = global_mean_pool(current_features, pool_batch_idx, size=B * M_out)
            
            Guiding_Poses_Mat_flat = se3_exp_map_custom(flat_guiding_poses_alg)
            R_guiding_flat = Guiding_Poses_Mat_flat[:, :3, :3]

            if torch.isnan(R_guiding_flat).any() or torch.isinf(R_guiding_flat).any():
                print("\n" + "#"*70)
                print("!!! EVIDENCE FOUND: The R_matrix input to D_from_matrix contains NaN/Inf !!!")

                problematic_poses_mask = torch.isnan(R_guiding_flat).any(dim=(1,2)) | torch.isinf(R_guiding_flat).any(dim=(1,2))
                problematic_input_poses = flat_guiding_poses_alg[problematic_poses_mask]

                print(f"    - Number of problematic poses: {problematic_input_poses.shape[0]}")
                print("    - This confirms that se3_exp_map_custom produced an invalid rotation matrix,")
                print("      likely due to extremely large values in its input 'guiding_poses_algebra'.")

                save_path = f"debug_tensors/problematic_guiding_poses_algebra.pt"
                torch.save(flat_guiding_poses_alg.detach().cpu(), save_path)
                print(f"    - Saved the FULL input tensor to D_from_matrix to: {save_path}")
                print("#"*70 + "\n")

                raise RuntimeError("Invalid R_matrix detected before calling D_from_matrix.")
            
            gconv_final_features = pooled_features
            if self.irreps_node_output_obj.lmax > 0:
                D_R_guiding = self._get_D_from_matrix_stable(self.irreps_node_output_obj, R_guiding_flat)
                gconv_final_features = torch.bmm(D_R_guiding, pooled_features.unsqueeze(-1)).squeeze(-1)
            
            gconv_final_features = gconv_final_features.reshape(B, M_out, -1)
            
            return gconv_final_features


    def __repr__(self):
        return f"{self.__class__.__name__}(in={self.irreps_node_input_obj}, out={self.irreps_node_output_obj})"

if __name__ == '__main__':
    import unittest

    class TestDifferentiableStability(unittest.TestCase):
        def setUp(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 实例化一个临时的 SE3GroupConvLayer 以便调用其方法
            self.gconv_layer_for_test = SE3GroupConvLayer(
                irreps_node_input="1x0e",
                irreps_node_output="1x1o+1x2o", # 包含奇偶宇称以触发 PowBackward2
                irreps_sh="1x0e",
                num_basis_radial=4,
                radial_mlp_hidden_dims=[]
            ).to(self.device)
            
            # 定义一个包含奇偶宇称的 Irreps 对象
            self.test_irreps = o3.Irreps("1x0e+1x1o+1x2e") # 1 + 3 + 5 = 9 dim

        def _create_bad_matrices(self, batch_size=4):
            """生成一批“坏”的、有瑕疵的3x3矩阵"""
            # 1. 完美的旋转矩阵
            perfect_R = o3.rand_matrix(batch_size, device=self.device)
            
            # 2. 引入瑕疵：非正交性
            noise = torch.randn(batch_size, 3, 3, device=self.device) * 0.1 # 增加噪声
            non_orthogonal_R = perfect_R + noise
            
            # 3. 引入瑕疵：缩放 (行列式 != 1)
            scales = torch.tensor([0.5, 1.0, 1.5, 5.0], device=self.device).view(-1, 1, 1)
            scaled_R = perfect_R * scales
            
            # 4. 引入瑕疵：反射 (行列式 = -1)
            reflection_matrix = torch.eye(3, device=self.device)
            reflection_matrix[2, 2] = -1
            reflected_R = perfect_R @ reflection_matrix
            
            # 5. 引入瑕疵：接近奇异的矩阵
            near_singular_R = torch.randn(batch_size, 3, 3, device=self.device)
            near_singular_R[:, 0, :] *= 1e-6 # 使其接近奇异
            
            bad_matrices = torch.cat([
                non_orthogonal_R, scaled_R, reflected_R, near_singular_R
            ], dim=0)
            
            bad_matrices.requires_grad_(True)
            return bad_matrices

        def test_differentiability_and_stability(self):
            """
            测试 _get_D_from_matrix_final 在处理“坏”矩阵时
            是否既能保持数值稳定，又能正确传播梯度。
            """
            print("\n--- Strict Test for _get_D_from_matrix_final ---")
            
            # 1. 生成一批有瑕疵的输入矩阵
            bad_R_matrices = self._create_bad_matrices()
            print(f"Testing with a batch of {bad_R_matrices.shape[0]} 'bad' matrices.")
            
            try:
                # 2. 通过我们的最终修复函数计算 D 矩阵
                D_matrices = self.gconv_layer_for_test._get_D_from_matrix_final(
                    self.test_irreps, bad_R_matrices
                )
                
                # 3. 检查前向传播的数值稳定性
                self.assertFalse(torch.isnan(D_matrices).any(), "Forward pass produced NaN in D_matrices.")
                self.assertFalse(torch.isinf(D_matrices).any(), "Forward pass produced Inf in D_matrices.")
                print("Forward pass is numerically stable (no NaNs/Infs).")

                # 4. 模拟损失计算和反向传播
                loss = D_matrices.sum()
                loss.backward()
                
                # 5. 检查梯度的可微性和稳定性
                grad = bad_R_matrices.grad
                self.assertIsNotNone(grad, "Gradient is None. Differentiability failed.")
                self.assertFalse(torch.isnan(grad).any(), "Backward pass produced NaN in gradients.")
                self.assertFalse(torch.isinf(grad).any(), "Backward pass produced Inf in gradients.")
                print("Backward pass is stable (no NaNs/Infs in gradient).")
                
                grad_norm = grad.norm().item()
                print(f"Gradient norm of input matrices: {grad_norm:.4f}")
                self.assertGreater(grad_norm, 1e-8, "Gradient is effectively zero.")
                
                print("\nSUCCESS: _get_D_from_matrix_final is both STABLE and DIFFERENTIABLE under stress.")

            except RuntimeError as e:
                # 如果代码在这里失败，测试就失败了
                self.fail(f"Test failed with a RuntimeError: {e}")
            except Exception as e:
                self.fail(f"Test failed with an unexpected exception: {e}")
                
    # 运行测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)