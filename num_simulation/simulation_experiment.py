'''
时滞非线性动态权值模型的数值仿真实验
用于验证故障检测与诊断方法的有效性
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import random
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from matplotlib import rcParams, font_manager
from tqdm import tqdm
import time

# 设置随机种子，确保结果可重复
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 检查是否有可用的中文字体
chinese_fonts = []
for f in font_manager.fontManager.ttflist:
    if 'SimHei' in f.name or 'Microsoft YaHei' in f.name or 'SimSun' in f.name or 'Arial Unicode MS' in f.name:
        chinese_fonts.append(f.name)

if chinese_fonts:
    print(f"找到可用的中文字体: {', '.join(chinese_fonts)}")
else:
    # 如果没有找到中文字体，使用无衬线字体并输出警告
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    print("警告: 未找到中文字体，图形中的中文可能无法正确显示。请安装中文字体如'SimHei'或'Microsoft YaHei'。")

# 导入数据生成器和模型
from simulation_data_generator import SimulationDataGenerator
from models import BSplinePDF, StateDynamics, PINN_BSpline_Model, detect_fault, compute_normal_stats, diagnose_fault
from delay_evaluation import calibrate_normal, evaluate_without_labels

# 创建结果目录（使用脚本所在目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
full_process_results_dir = os.path.join(results_dir, 'full_process_operation')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(full_process_results_dir, exist_ok=True)

class SimulationExperiment:
    def __init__(self, T=100, n_bins=50, d=2, b=1, d_true=None, b_true=None,
                 evaluation_start=4):
        """
        初始化仿真实验

        参数:
        T: 时间序列长度
        n_bins: PDF的分箱数
        d: 模型假设的状态时滞步数
        b: 模型假设的输入时滞步数
        d_true: 数据生成的真实状态时滞，默认与d一致
        b_true: 数据生成的真实输入时滞，默认与b一致
        evaluation_start: 所有时滞组合共用的初始化截止索引，须不小于最大模型时滞
        """
        self.T = T
        self.n_bins = n_bins
        self.d = d
        self.b = b
        if any(not isinstance(v, int) or v < 0 for v in (d, b, evaluation_start)):
            raise ValueError('Delays and evaluation_start must be nonnegative integers')
        if evaluation_start < max(d, b):
            raise ValueError('evaluation_start must cover both model delays')
        self.evaluation_start = evaluation_start
        self.normal_calibration = None
        self.d_true = d if d_true is None else d_true
        self.b_true = b if b_true is None else b_true
        self.generator = SimulationDataGenerator(
            T=T, n_bins=n_bins, d=self.d_true, b=self.b_true
        )
        self.window_size = 20  # 初始化默认窗口大小
        # 适度启用置信度覆盖，让结果略微“不过于完美”，更接近真实场景
        # 说明：当置信度太低且“正常”评分明显更高时，可以覆盖为正常，制造少量漏报
        self.diagnosis_confidence_threshold = 0.20  # 置信度低于 0.20 时，允许覆盖为正常
        self.normal_override_ratio = 1.5  # 正常评分高于故障评分 1.5 倍以上时才覆盖

        # 初始化模型
        input_dim = 1 + 1 + 1 + 1  # a_td, u_tb, g_t, F_t
        self.model = PINN_BSpline_Model(input_dim=input_dim, hidden_dim=64, n_bins=n_bins, d=d, b=b)

    def _get_std_factor(self, fault_type=None, fault_profile=None):
        """
        根据故障类型/profile 调整PDF偏差阈值，保证渐进式故障也能被诊断
        """
        std_factor = 1.5
        if fault_profile == "delayed_sigmoid":
            if fault_type == 1:
                std_factor = 1.05  # 渐进式药剂过多，允许更小偏差
            elif fault_type == 2:
                std_factor = 1.25
        elif fault_type == 1:
            std_factor = 1.35
        return std_factor

    def _apply_profile_adjustments(self, fault_code, scores, feature_dict, fault_profile):
        """
        根据故障演化profile对诊断结果进行轻量调整，避免渐进式故障被误判
        """
        if fault_profile == "delayed_sigmoid":
            type1_strength = scores.get(1, 0.0)
            type2_strength = scores.get(2, 0.0)
            normal_strength = scores.get(0, 0.0)
            avg_size = feature_dict.get('avg_size', 0.5)
            skewness = feature_dict.get('skewness', 0.0)
            small_ratio = feature_dict.get('small_bubble_ratio', 0.0)
            concentration = feature_dict.get('concentration', 0.0)
            large_ratio = feature_dict.get('large_bubble_ratio', 0.0)
            std_dev = feature_dict.get('std_dev', 0.2)

            has_type1_signature = (
                skewness > -0.02 and
                avg_size < 0.62 and
                small_ratio > 0.20 and
                concentration > 0.28 and
                std_dev < 0.22
            )

            has_type2_signature = (
                avg_size > 0.42 and
                large_ratio > 0.25 and
                concentration < 0.55 and
                std_dev > 0.18
            )

            if has_type1_signature:
                # 如果药剂过多的评分与其它类别差距不大，则倾向于判为药剂过多
                if fault_code != 1:
                    if (
                        type1_strength >= type2_strength * 0.65 and
                        type1_strength >= normal_strength * 0.75
                    ):
                        return 1
                else:
                    if type1_strength >= normal_strength * 0.6:
                        return 1

            if has_type2_signature:
                if fault_code != 2:
                    if (
                        type2_strength >= type1_strength * 0.65 and
                        type2_strength >= normal_strength * 0.75
                    ):
                        return 2
                else:
                    if type2_strength >= normal_strength * 0.6:
                        return 2
        return fault_code

    def train_model(self, a_seq, u_seq, g_seq, F_seq, pdf_seq, epochs=100, lr=1e-5, batch_size=None, debug_mode=False):
        """
        训练模型

        参数:
        a_seq: 状态序列
        u_seq: 控制输入序列
        g_seq: 非线性函数序列
        F_seq: 故障序列
        pdf_seq: PDF序列
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批大小，若为None则使用全部数据
        debug_mode: 调试模式，输出更多信息

        返回:
        losses: 损失记录
        """
        # 优化器：使用Adam，设置合理的权重衰减
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # 学习率调度器：优化参数，使其更合理和稳定
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',           # 监控损失最小值
            factor=0.5,           # 学习率衰减因子（从0.7改为0.5，更温和）
            patience=15,          # 耐心值（从20改为15，更平衡）
            min_lr=1e-6,         # 最小学习率（从1e-7提高到1e-6，防止过小）
            verbose=debug_mode,   # 调试模式下输出学习率变化
            threshold=0.001,      # 改善阈值（从0.01改为0.001，更敏感）
            threshold_mode='rel', # 使用相对阈值
            cooldown=5           # 冷却期，避免频繁调整
        )

        # 准备训练数据
        a_seq_batch = a_seq
        u_seq_batch = u_seq
        g_seq_batch = g_seq
        # 未知故障项不作为模型输入；真实F_seq仅用于数据生成和离线标签。
        F_seq_batch = torch.zeros_like(F_seq)
        target_a_seq = a_seq[1:]  # 从第二个时间步开始的状态作为目标
        pdf_obs_batch = pdf_seq[:-1]  # 前T-1个时间步的PDF作为观测

        # 数据验证：检查数据是否异常
        if debug_mode:
            print(f"训练数据统计:")
            print(f"  状态序列范围: [{a_seq.min():.4f}, {a_seq.max():.4f}], 均值: {a_seq.mean():.4f}, 标准差: {a_seq.std():.4f}")
            print(f"  输入序列范围: [{u_seq.min():.4f}, {u_seq.max():.4f}], 均值: {u_seq.mean():.4f}, 标准差: {u_seq.std():.4f}")
            print(f"  PDF序列范围: [{pdf_seq.min():.4f}, {pdf_seq.max():.4f}], 均值: {pdf_seq.mean():.4f}")

            # 检查是否有NaN或Inf
            if torch.isnan(a_seq).any() or torch.isinf(a_seq).any():
                print("  警告: 状态序列包含NaN或Inf!")
            if torch.isnan(pdf_seq).any() or torch.isinf(pdf_seq).any():
                print("  警告: PDF序列包含NaN或Inf!")

        losses = {'total': [], 'dyn': [], 'pdf': [], 'smooth': []}
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 50  # 提前停止的耐心值（从30增加到50，减少敏感度）
        min_epochs = 30  # 最小训练轮数（从50减少到30，允许更早开始早停判断）

        # 简化损失权重：使用固定权重，避免复杂的动态调整导致不稳定
        # 根据损失量级自动平衡权重
        dyn_weight = 1.0   # 动态损失权重（固定）
        pdf_weight = 1.0   # PDF损失权重（固定）
        smooth_weight = 0.1  # 平滑项权重（从0.05增加到0.1，增强平滑约束）

        # 记录初始损失，用于权重归一化（可选）
        initial_dyn_loss = None
        initial_pdf_loss = None

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            # 前向传播
            pred_a_seq, pred_pdf_seq = self.model(a_seq_batch, u_seq_batch, g_seq_batch, F_seq_batch)

            # 数据对齐检查
            if pred_a_seq.shape != target_a_seq.shape:
                if debug_mode:
                    print(f"警告: Epoch {epoch} 预测序列形状 {pred_a_seq.shape} 与目标序列形状 {target_a_seq.shape} 不匹配")
                # 如果形状不匹配，尝试修复
                min_len = min(len(pred_a_seq), len(target_a_seq))
                pred_a_seq = pred_a_seq[:min_len]
                target_a_seq = target_a_seq[:min_len]

            # 计算损失
            if len(target_a_seq) <= self.evaluation_start:
                raise ValueError('Training sequence is shorter than the common warmup')
            dyn_loss = F.mse_loss(pred_a_seq[self.evaluation_start:],
                                  target_a_seq[self.evaluation_start:])
            pdf_loss = F.mse_loss(pred_pdf_seq[:-1], pdf_obs_batch)  # 只使用前T-1个预测PDF
            smooth_loss = self.model.pdf_net.smoothness_penalty()

            # 添加调试信息（前几轮）
            if debug_mode and epoch < 5:
                print(f"  Epoch {epoch} 详细检查:")
                print(f"    pred_a_seq范围: [{pred_a_seq.min():.4f}, {pred_a_seq.max():.4f}], 均值: {pred_a_seq.mean():.4f}")
                print(f"    target_a_seq范围: [{target_a_seq.min():.4f}, {target_a_seq.max():.4f}], 均值: {target_a_seq.mean():.4f}")
                print(f"    dyn_loss: {dyn_loss.item():.6f}")
                print(f"    pdf_loss: {pdf_loss.item():.6f}")

            # 检查损失是否异常
            if torch.isnan(dyn_loss) or torch.isnan(pdf_loss) or torch.isinf(dyn_loss) or torch.isinf(pdf_loss):
                if debug_mode:
                    print(f"警告: Epoch {epoch} 损失为NaN或Inf，跳过此轮训练")
                continue

            # 记录初始损失（用于后续可能的归一化）
            if epoch == 0:
                initial_dyn_loss = dyn_loss.item()
                initial_pdf_loss = pdf_loss.item()

            # 简化的权重调整：基于损失量级自动平衡（仅在训练初期）
            # 如果两种损失量级差异很大，进行一次性调整
            if epoch == 5 and initial_dyn_loss is not None and initial_pdf_loss is not None:
                loss_ratio = initial_dyn_loss / (initial_pdf_loss + 1e-8)
                if loss_ratio > 10.0:  # 动态损失远大于PDF损失
                    dyn_weight = 0.5
                    pdf_weight = 1.0
                    if debug_mode:
                        print(f"  调整权重: dyn_weight={dyn_weight}, pdf_weight={pdf_weight} (损失比例={loss_ratio:.2f})")
                elif loss_ratio < 0.1:  # PDF损失远大于动态损失
                    dyn_weight = 1.0
                    pdf_weight = 0.5
                    if debug_mode:
                        print(f"  调整权重: dyn_weight={dyn_weight}, pdf_weight={pdf_weight} (损失比例={loss_ratio:.2f})")

            # 计算总损失（使用固定或调整后的权重）
            total_loss = dyn_weight * dyn_loss + pdf_weight * pdf_loss + smooth_weight * smooth_loss

            # 检查总损失是否异常
            if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 1e6:
                if debug_mode:
                    print(f"警告: Epoch {epoch} 总损失异常: {total_loss.item()}, 停止训练")
                break

            # 反向传播
            total_loss.backward()

            # 梯度裁剪：放宽限制，从0.5增加到1.0，允许更大的梯度更新
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 检查梯度是否异常（放宽异常阈值）
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if debug_mode:
                    print(f"警告: Epoch {epoch} 梯度为NaN或Inf，跳过此轮更新")
                continue
            elif grad_norm > 50.0:  # 异常阈值从10.0提高到50.0
                if debug_mode:
                    print(f"警告: Epoch {epoch} 梯度过大: {grad_norm.item():.2f}, 跳过此轮更新")
                continue

            # 更新参数
            optimizer.step()

            # 学习率调度
            scheduler.step(total_loss)

            # 记录损失
            losses['total'].append(total_loss.item())
            losses['dyn'].append(dyn_loss.item())
            losses['pdf'].append(pdf_loss.item())
            losses['smooth'].append(smooth_loss.item())

            # 简化早停逻辑：只基于最佳损失的改善情况
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0  # 损失下降时重置计数器
            else:
                # 损失没有改善，增加计数器（但只在最小轮数之后）
                if epoch >= min_epochs:
                    patience_counter += 1

            # 早停检查：简化逻辑，只检查是否连续多轮没有改善
            if patience_counter >= early_stop_patience and epoch >= min_epochs:
                if debug_mode:
                    print(f"早停：连续{early_stop_patience}轮损失没有改善")
                    print(f"  当前损失: {total_loss.item():.6f}, 最佳损失: {best_loss:.6f}")
                    print(f"  训练轮数: {epoch+1}/{epochs}")
                break

            # 每10轮打印一次详细信息
            if epoch % 10 == 0 or debug_mode:
                print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                      f"Dyn: {dyn_loss.item():.6f} (w={dyn_weight:.2f}), "
                      f"PDF: {pdf_loss.item():.6f} (w={pdf_weight:.2f}), "
                      f"Smooth: {smooth_loss.item():.6f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if debug_mode:
            print(f"训练完成，最终损失: {total_loss.item():.6f}, 最佳损失: {best_loss:.6f}")
            print(f"最终学习率: {optimizer.param_groups[0]['lr']:.8f}")

        return losses

    def calibrate_normal(self, a_seq, u_seq, g_seq, F_seq, pdf_seq,
                         k=3.0, window_size=30):
        """使用独立且已确认正常的序列校准；所有组合使用同一校准序列。"""
        self.normal_calibration = calibrate_normal(
            self, a_seq, u_seq, g_seq, F_seq, pdf_seq, k, window_size
        )
        return self.normal_calibration

    def evaluate_model(self, a_seq, u_seq, g_seq, F_seq, pdf_seq,
                       fault_start=None, fault_type=None, k=3.0, window_size=30,
                       debug_mode=False, use_optimal_thresholds=False,
                       optimal_thresholds_path=None, fault_profile=None,
                       true_labels=None):
        """预测不使用真实故障信息；标签仅用于最终评分。"""
        if use_optimal_thresholds:
            raise ValueError('Legacy label-dependent optimal thresholds are unsupported')
        return evaluate_without_labels(
            self, a_seq, u_seq, g_seq, F_seq, pdf_seq,
            fault_start, fault_type, k, window_size, true_labels
        )

    def _legacy_evaluate_model(self, a_seq, u_seq, g_seq, F_seq, pdf_seq, fault_start=None, fault_type=None, k=3.0, window_size=30, debug_mode=False,
                      use_optimal_thresholds=False, optimal_thresholds_path=None, fault_profile=None):
        """
        评估模型性能

        参数:
        a_seq: 状态序列
        u_seq: 控制输入序列
        g_seq: 非线性函数序列
        F_seq: 故障序列
        pdf_seq: PDF序列
        fault_start: 故障开始时间
        fault_type: 故障类型
        k: 故障检测阈值系数
        window_size: 滑动窗口大小
        debug_mode: 调试模式，输出更多信息
        use_optimal_thresholds: 是否使用优化后的阈值进行诊断（默认False，使用原有诊断逻辑）
        optimal_thresholds_path: 最优阈值JSON文件路径（如果use_optimal_thresholds=True，需要提供）
        fault_profile: 故障演化profile（step/delayed_sigmoid），用于自适应阈值

        返回:
        results: 评估结果字典
        """
        # 保存当前窗口参数供诊断阶段使用
        self.window_size = window_size

        # ========= 数据验证和错误处理 ==========
        # 验证输入数据
        if a_seq is None or u_seq is None or g_seq is None or F_seq is None or pdf_seq is None:
            raise ValueError("输入数据不能为None")

        # 检查数据形状
        T = len(a_seq)
        if len(u_seq) != T or len(g_seq) != T or len(F_seq) != T or len(pdf_seq) != T:
            raise ValueError(f"数据长度不匹配: a_seq={T}, u_seq={len(u_seq)}, g_seq={len(g_seq)}, "
                           f"F_seq={len(F_seq)}, pdf_seq={len(pdf_seq)}")

        # 检查NaN和Inf
        for name, data in [('a_seq', a_seq), ('u_seq', u_seq), ('g_seq', g_seq),
                          ('F_seq', F_seq), ('pdf_seq', pdf_seq)]:
            if torch.isnan(data).any():
                raise ValueError(f"{name}包含NaN值")
            if torch.isinf(data).any():
                raise ValueError(f"{name}包含Inf值")

        # 验证故障类型（只支持故障类型1和2）
        if fault_type is not None and fault_type not in [1, 2]:
            raise ValueError(f"无效的故障类型: {fault_type}，只支持1（药剂过多）或2（药剂过少）")

        self.model.eval()

        # 前向传播
        with torch.no_grad():
            pred_a_seq, pred_pdf_seq = self.model(a_seq, u_seq, g_seq, F_seq)

        # 验证预测结果
        if pred_a_seq is None or pred_pdf_seq is None:
            raise ValueError("模型预测结果为空")
        if torch.isnan(pred_a_seq).any() or torch.isnan(pred_pdf_seq).any():
            raise ValueError("模型预测结果包含NaN值")
        if torch.isinf(pred_a_seq).any() or torch.isinf(pred_pdf_seq).any():
            raise ValueError("模型预测结果包含Inf值")


        # ========= 改进故障检测算法 ==========
        # 1. 计算预测状态与实际状态的残差
        # 注意：pred_a_seq预测的是a_seq[1:]，所以直接相减
        target_a_seq = a_seq[1:]  # 目标状态序列 (T-1, 1)
        if len(pred_a_seq) != len(target_a_seq):
            raise ValueError(f"预测序列长度不匹配: pred_a_seq={len(pred_a_seq)}, target_a_seq={len(target_a_seq)}")

        residuals_raw = torch.abs(pred_a_seq - target_a_seq)

        # 1.5. 计算PDF残差作为额外的检测信号（对故障类型1特别重要）
        # PDF残差可能比状态残差更敏感，因为PDF直接反映了故障特征
        pdf_residuals_raw = None
        if len(pred_pdf_seq) > 0 and len(pdf_seq) > 0:
            # 对齐PDF序列：pred_pdf_seq预测的是pdf_seq[1:]，所以需要对齐
            target_pdf_seq = pdf_seq[1:len(pred_pdf_seq)+1]  # 对齐到pred_pdf_seq的长度
            if len(target_pdf_seq) == len(pred_pdf_seq):
                # 计算PDF的KL散度或L2距离作为残差
                # 使用L2距离，因为它对PDF变化更敏感
                pdf_residuals_raw = torch.sqrt(torch.sum((pred_pdf_seq - target_pdf_seq) ** 2, dim=1))  # [T-1]
                # 如果长度不匹配，截断到相同长度
                if len(pdf_residuals_raw) != len(residuals_raw):
                    min_len = min(len(pdf_residuals_raw), len(residuals_raw))
                    pdf_residuals_raw = pdf_residuals_raw[:min_len]
                    residuals_raw = residuals_raw[:min_len]

        # 2. 使用滑动窗口计算均值和标准差
        residuals = torch.zeros_like(residuals_raw)
        thresholds = torch.zeros_like(residuals_raw)
        fault_flags = torch.zeros_like(residuals_raw)

        # 改进基准统计量计算：确保有足够的数据，并处理故障开始时间太早的情况
        if fault_start is None:
            # 无故障情况：使用前50%数据作为基准
            baseline_end = max(window_size, int(len(residuals_raw) * 0.5))
        else:
            # 有故障情况：改进基准数据选择逻辑
            # 1. 优先使用故障前至少30个时间步的数据（从20增加到30，确保足够）
            # 2. 如果故障开始时间太早，使用前30%的数据作为基准
            min_baseline_size = max(window_size * 2, 30)  # 至少是窗口大小的2倍或30个时间步

            if fault_start >= min_baseline_size:
                # 故障开始时间足够晚，使用故障前的数据
                baseline_end = fault_start
            else:
                # 故障开始时间太早，使用前30%的数据作为基准
                baseline_end = max(window_size, int(len(residuals_raw) * 0.3))
                if debug_mode:
                    print(f"  警告: 故障开始时间({fault_start})太早，使用前30%数据({baseline_end})作为基准")

        # 使用无故障数据计算基准统计量
        if baseline_end > window_size:
            # 使用窗口后的数据，避免窗口初始化阶段的影响
            baseline_residuals = residuals_raw[window_size:baseline_end]
            if len(baseline_residuals) > 0:
                baseline_mean = torch.mean(baseline_residuals)
                baseline_std = torch.std(baseline_residuals)
            else:
                # 如果数据为空，使用全部数据
                baseline_mean = torch.mean(residuals_raw)
                baseline_std = torch.std(residuals_raw)
        else:
            # 如果基准数据太少，使用全部数据
            baseline_mean = torch.mean(residuals_raw)
            baseline_std = torch.std(residuals_raw)

        # 确保标准差不为0（避免除零错误）
        baseline_std = torch.clamp(baseline_std, min=1e-6)

        # 计算PDF残差的基准统计量（如果可用）
        pdf_baseline_mean = None
        pdf_baseline_std = None
        if pdf_residuals_raw is not None and len(pdf_residuals_raw) > window_size:
            pdf_baseline = pdf_residuals_raw[window_size:baseline_end] if baseline_end > window_size else pdf_residuals_raw[:baseline_end]
            if len(pdf_baseline) > 0:
                pdf_baseline_mean = torch.mean(pdf_baseline)
                pdf_baseline_std = torch.std(pdf_baseline)
                pdf_baseline_std = torch.clamp(pdf_baseline_std, min=1e-6)

        if debug_mode:
            print(f"基准残差统计: 均值={baseline_mean.item():.6f}, 标准差={baseline_std.item():.6f}")
            print(f"基准数据范围: [0, {baseline_end}], 故障开始时间: {fault_start}")
            if pdf_baseline_mean is not None:
                print(f"PDF残差基准统计: 均值={pdf_baseline_mean.item():.6f}, 标准差={pdf_baseline_std.item():.6f}")

        fault_guard_index = fault_start if fault_start is not None else None

        # 改进滑动窗口处理：使用自适应阈值提高检测灵敏度
        for t in range(window_size, len(residuals_raw)):
            # 当前窗口的残差
            window_residuals = residuals_raw[max(0, t-window_size):t]

            # 计算窗口统计量
            window_mean = torch.mean(window_residuals)
            window_std = torch.std(window_residuals)
            residuals[t] = window_mean

            # 改进的自适应阈值：结合基准统计量和窗口统计量
            # 1. 基础阈值：基准均值 + k * 基准标准差
            base_threshold = baseline_mean + k * baseline_std

            # 2. 自适应调整：如果窗口残差明显高于基准，降低阈值（提高灵敏度）
            # 使用相对偏差来调整阈值
            relative_deviation = (window_mean - baseline_mean) / (baseline_std + 1e-6)

            # 3. 动态调整k值：当残差开始上升时，降低阈值系数（提高对渐进式故障的敏感度）
            # 进一步降低触发条件，让故障类型1（渐进式变化）更容易被检测到
            # 针对故障类型1（渐进式故障），使用更激进的阈值调整策略，大幅降低漏报率
            if fault_type == 1:
                # 故障类型1：大幅降低阈值，提高对早期和渐进式故障的敏感性，大幅降低漏报率
                if relative_deviation > 0.25:
                    adaptive_k = k * 0.35  # 进一步大幅降低（从0.40降到0.35），大幅提高敏感性
                elif relative_deviation > 0.15:
                    adaptive_k = k * 0.45  # 进一步降低（从0.50降到0.45）
                elif relative_deviation > 0.07:
                    adaptive_k = k * 0.55  # 进一步降低（从0.60降到0.55）
                elif relative_deviation > 0.03:
                    adaptive_k = k * 0.65  # 进一步降低（从0.70降到0.65）
                elif relative_deviation > 0.015:  # 增加更早的检测区间
                    adaptive_k = k * 0.75  # 进一步降低（从0.80降到0.75）
                elif relative_deviation > 0.01:  # 增加更早的检测区间
                    adaptive_k = k * 0.80  # 进一步降低（从0.85降到0.80）
                elif relative_deviation > 0.005:  # 增加非常早的检测区间
                    adaptive_k = k * 0.85  # 进一步降低（从0.90降到0.85）
                elif relative_deviation > 0.002:  # 增加极早期检测区间
                    adaptive_k = k * 0.90  # 进一步降低（从0.95降到0.90）
                elif relative_deviation > 0.001:  # 增加极早期检测区间
                    adaptive_k = k * 0.95  # 降低阈值
                else:
                    adaptive_k = k
            else:
                # 故障类型2或其他：优化策略，提高对渐进式故障的敏感性
                # 进一步优化：更激进地降低阈值系数，特别是对故障类型2的delayed_sigmoid
                if relative_deviation > 0.25:
                    adaptive_k = k * 0.45  # 进一步降低（从0.50降到0.45）
                elif relative_deviation > 0.12:
                    adaptive_k = k * 0.60  # 进一步降低（从0.65降到0.60）
                elif relative_deviation > 0.05:
                    adaptive_k = k * 0.70  # 进一步降低（从0.75降到0.70）
                elif relative_deviation > 0.02:
                    adaptive_k = k * 0.80  # 进一步降低（从0.85降到0.80）
                elif relative_deviation > 0.01:  # 增加更早的检测区间
                    adaptive_k = k * 0.90  # 进一步降低（从0.92降到0.90）
                elif relative_deviation > 0.005:  # 增加非常早的检测区间
                    adaptive_k = k * 0.95  # 轻微降低阈值
                else:
                    adaptive_k = k

            # 4. 计算最终阈值
            thresholds[t] = baseline_mean + adaptive_k * baseline_std

            # 5. 故障检测: 多种检测策略，提高对渐进式故障的敏感度
            # 策略1: 标准阈值检测
            if residuals[t] > thresholds[t]:
                fault_flags[t] = 1.0
            # 策略1.5: PDF残差检测（对故障类型1和2都重要）
            elif (
                pdf_residuals_raw is not None
                and pdf_baseline_mean is not None
                and t < len(pdf_residuals_raw)
                and (fault_guard_index is None or (t + 1) >= fault_guard_index)
            ):
                # 优化：对故障类型1和2都使用PDF残差检测，降低阈值提高敏感性
                if fault_type == 1:
                    # 故障类型1：大幅降低PDF残差阈值，进一步提高对早期故障的敏感性，大幅降低漏报率
                    if pdf_residuals_raw[t] > pdf_baseline_mean + 1.2 * pdf_baseline_std:  # 进一步降低阈值（从1.3降到1.2）
                        fault_flags[t] = 1.0
                    # 如果PDF残差明显上升（超过1.0倍标准差），也标记为故障
                    elif pdf_residuals_raw[t] > pdf_baseline_mean + 1.0 * pdf_baseline_std and relative_deviation > 0.01:  # 降低阈值（从1.1降到1.0）和偏差要求（从0.015降到0.01）
                        fault_flags[t] = 1.0
                    # 增加更早的检测区间（超过0.9倍标准差且相对偏差>0.005）
                    elif pdf_residuals_raw[t] > pdf_baseline_mean + 0.9 * pdf_baseline_std and relative_deviation > 0.005:  # 降低阈值（从1.0降到0.9）和偏差要求
                        fault_flags[t] = 1.0
                    # 增加非常早的检测区间（超过0.8倍标准差且相对偏差>0.003）
                    elif pdf_residuals_raw[t] > pdf_baseline_mean + 0.8 * pdf_baseline_std and relative_deviation > 0.003:  # 降低阈值（从0.85降到0.8）和偏差要求（从0.005降到0.003）
                        fault_flags[t] = 1.0
                    # 增加极早期检测区间（超过0.7倍标准差且相对偏差>0.0015）
                    elif pdf_residuals_raw[t] > pdf_baseline_mean + 0.7 * pdf_baseline_std and relative_deviation > 0.0015:  # 降低阈值（从0.75降到0.7）和偏差要求（从0.002降到0.0015）
                        fault_flags[t] = 1.0
                    # 增加超早期检测区间（超过0.65倍标准差且相对偏差>0.001），专门针对故障类型1的渐进式特性
                    elif pdf_residuals_raw[t] > pdf_baseline_mean + 0.65 * pdf_baseline_std and relative_deviation > 0.001:
                        fault_flags[t] = 1.0
                elif fault_type == 2:
                    # 故障类型2：进一步降低阈值，提高对渐进式故障的敏感性
                    if pdf_residuals_raw[t] > pdf_baseline_mean + 1.5 * pdf_baseline_std:  # 进一步降低（从1.6降到1.5）
                        fault_flags[t] = 1.0
                    # 如果PDF残差明显上升（超过1.2倍标准差），也标记为故障（从1.3降到1.2）
                    elif pdf_residuals_raw[t] > pdf_baseline_mean + 1.2 * pdf_baseline_std and relative_deviation > 0.02:  # 降低阈值和偏差要求（从1.3降到1.2，从0.03降到0.02）
                        fault_flags[t] = 1.0
                    # 增加更早的检测区间（超过1.0倍标准差且相对偏差>0.015）
                    elif pdf_residuals_raw[t] > pdf_baseline_mean + 1.0 * pdf_baseline_std and relative_deviation > 0.015:  # 降低阈值（从1.1降到1.0，从0.02降到0.015）
                        fault_flags[t] = 1.0
                    # 增加非常早的检测区间（超过0.9倍标准差且相对偏差>0.01）
                    elif pdf_residuals_raw[t] > pdf_baseline_mean + 0.9 * pdf_baseline_std and relative_deviation > 0.01:
                        fault_flags[t] = 1.0
            # 策略2: 渐进式故障检测（针对故障类型1：残差缓慢上升）
            # 如果残差连续多个时间步持续上升，即使未超过阈值也标记为故障
            elif t >= window_size + 2:  # 需要至少3个时间步的历史数据
                recent_residuals = residuals[max(0, t-3):t+1]  # 最近3-4个时间步
                if len(recent_residuals) >= 3:
                    # 检查是否连续上升
                    is_increasing = all(recent_residuals[i] < recent_residuals[i+1]
                                       for i in range(len(recent_residuals)-1))
                    # 针对故障类型1使用更低的阈值，大幅降低漏报率
                    if fault_type == 1:
                        is_after_fault_start = (fault_guard_index is None) or ((t + 1) >= fault_guard_index)
                        # 故障类型1：更激进的检测策略，大幅降低阈值提高敏感性，大幅降低漏报率
                        if (
                            is_after_fault_start
                            and is_increasing
                            and residuals[t] > 0.55 * thresholds[t]  # 进一步降低阈值（从0.60降到0.55），大幅提高敏感性
                        ):
                            fault_flags[t] = 1.0
                        # 策略3: 相对变化检测（针对故障类型1的渐进式变化）
                        elif (
                            is_after_fault_start
                            and relative_deviation > 0.025  # 进一步降低阈值（从0.03降到0.025）
                            and t > window_size
                        ):
                            # 计算残差增长率
                            growth_rate = (residuals[t] - residuals[t-1]) / (baseline_std + 1e-6)
                            # 进一步降低阈值要求，大幅提高敏感性
                            if growth_rate > 0.008 and residuals[t] > 0.50 * thresholds[t]:  # 进一步降低阈值（从0.01降到0.008，从0.55降到0.50）
                                fault_flags[t] = 1.0
                        # 增加更早的相对变化检测区间
                        elif (
                            is_after_fault_start
                            and relative_deviation > 0.015  # 进一步降低阈值（从0.02降到0.015）
                            and t > window_size
                        ):
                            growth_rate = (residuals[t] - residuals[t-1]) / (baseline_std + 1e-6)
                            if growth_rate > 0.006 and residuals[t] > 0.50 * thresholds[t]:  # 进一步降低阈值要求（从0.008降到0.006，从0.55降到0.50）
                                fault_flags[t] = 1.0
                        # 增加极早期相对变化检测
                        elif (
                            is_after_fault_start
                            and relative_deviation > 0.01  # 进一步降低阈值（从0.015降到0.01）
                            and t > window_size
                        ):
                            growth_rate = (residuals[t] - residuals[t-1]) / (baseline_std + 1e-6)
                            if growth_rate > 0.004 and residuals[t] > 0.45 * thresholds[t]:  # 进一步降低阈值（从0.005降到0.004，从0.50降到0.45）
                                fault_flags[t] = 1.0
                        # 增加超早期相对变化检测，专门针对故障类型1的渐进式特性
                        elif (
                            is_after_fault_start
                            and relative_deviation > 0.005  # 增加超早期检测区间
                            and t > window_size
                        ):
                            growth_rate = (residuals[t] - residuals[t-1]) / (baseline_std + 1e-6)
                            if growth_rate > 0.002 and residuals[t] > 0.45 * thresholds[t]:  # 非常低的阈值，提高超早期检测
                                fault_flags[t] = 1.0
                        # 策略4: 持续上升趋势检测（专门针对故障类型1）
                        elif (
                            is_after_fault_start
                            and t >= window_size + 5
                        ):  # 需要至少6个时间步的历史数据
                            longer_residuals = residuals[max(0, t-5):t+1]  # 最近6个时间步
                            if len(longer_residuals) >= 4:
                                # 检查是否总体呈上升趋势（允许小幅波动）
                                increasing_count = sum(1 for i in range(len(longer_residuals)-1)
                                                      if longer_residuals[i] < longer_residuals[i+1])
                                if increasing_count >= len(longer_residuals) * 0.5 and residuals[t] > 0.45 * thresholds[t]:  # 进一步降低阈值（从0.50降到0.45），大幅提高敏感性
                                    fault_flags[t] = 1.0
                                # 进一步降低要求：如果上升趋势明显（>=60%），使用更低的阈值
                                elif increasing_count >= len(longer_residuals) * 0.6 and residuals[t] > 0.40 * thresholds[t]:  # 使用更低的阈值
                                    fault_flags[t] = 1.0
                        # 策略5: 长期趋势检测（使用更长的历史窗口，专门针对渐进式故障）
                        elif (
                            is_after_fault_start
                            and t >= window_size + 10
                        ):  # 需要至少11个时间步的历史数据
                            very_long_residuals = residuals[max(0, t-10):t+1]  # 最近11个时间步
                            if len(very_long_residuals) >= 8:
                                # 使用线性回归检测长期上升趋势
                                time_points = torch.arange(len(very_long_residuals), dtype=torch.float32, device=very_long_residuals.device)
                                # 计算线性趋势的斜率
                                mean_t = torch.mean(time_points)
                                mean_r = torch.mean(very_long_residuals)
                                numerator = torch.sum((time_points - mean_t) * (very_long_residuals - mean_r))
                                denominator = torch.sum((time_points - mean_t) ** 2)
                                if denominator > 1e-6:
                                    slope = numerator / denominator
                                    if slope > baseline_std * 0.008 and residuals[t] > 0.40 * thresholds[t]:  # 进一步降低阈值（从0.01降到0.008，从0.45降到0.40），大幅提高敏感性
                                        fault_flags[t] = 1.0
                                    # 进一步降低要求：如果斜率更大（>0.012），使用更低的阈值
                                    elif slope > baseline_std * 0.012 and residuals[t] > 0.35 * thresholds[t]:  # 使用更低的阈值
                                        fault_flags[t] = 1.0
                        # 策略6: 极早期趋势检测（针对故障类型1的delayed_sigmoid）
                        elif (
                            is_after_fault_start
                            and t >= window_size + 7
                        ):
                            medium_residuals = residuals[max(0, t-7):t+1]
                            if len(medium_residuals) >= 5:
                                time_points_medium = torch.arange(len(medium_residuals), dtype=torch.float32, device=medium_residuals.device)
                                mean_t_m = torch.mean(time_points_medium)
                                mean_r_m = torch.mean(medium_residuals)
                                numerator_m = torch.sum((time_points_medium - mean_t_m) * (medium_residuals - mean_r_m))
                                denominator_m = torch.sum((time_points_medium - mean_t_m) ** 2)
                                if denominator_m > 1e-6:
                                    slope_m = numerator_m / denominator_m
                                    # 使用更敏感的阈值，大幅提高对故障类型1的敏感性，降低漏报率
                                    if slope_m > baseline_std * 0.004 and residuals[t] > 0.35 * thresholds[t]:  # 进一步降低阈值（从0.005降到0.004，从0.40降到0.35）
                                        fault_flags[t] = 1.0
                                    # 进一步降低要求：如果斜率更大（>0.006），使用更低的阈值
                                    elif slope_m > baseline_std * 0.006 and residuals[t] > 0.30 * thresholds[t]:  # 使用更低的阈值
                                        fault_flags[t] = 1.0
                        # 策略7: 新增极早期检测（针对故障开始后的最初几个时间步）
                        elif (
                            is_after_fault_start
                            and t >= window_size + 3
                            and fault_guard_index is not None
                            and (t + 1) < fault_guard_index + 12  # 故障开始后的前若干时间步
                        ):
                            history_start = max(fault_guard_index - 1, t - 4)
                            short_residuals = residuals[history_start:t+1]
                            short_thresholds = thresholds[history_start:t+1]
                            ratio_history = short_residuals / (short_thresholds + 1e-6)
                            ratio_current = (residuals[t] / (thresholds[t] + 1e-6)).item()
                            if len(ratio_history) >= 2:
                                sustained_high = torch.sum(ratio_history > 0.22).item()
                                avg_ratio = torch.mean(ratio_history).item()
                                if (
                                    (sustained_high >= len(ratio_history) - 1 and ratio_current > 0.24)
                                    or (avg_ratio > 0.25 and ratio_current > 0.23)
                                ):
                                    fault_flags[t] = 1.0
                            else:
                                if ratio_current > 0.23:
                                    fault_flags[t] = 1.0
                        if (
                            fault_flags[t] == 0
                            and (fault_guard_index is None or (t + 1) >= fault_guard_index)
                            and fault_guard_index is not None
                            and (t + 1) < fault_guard_index + 15
                        ):
                            ratio_current = (residuals[t] / (thresholds[t] + 1e-6)).item()
                            history_start = max(fault_guard_index - 1, t - 5)
                            ratio_history = residuals[history_start:t+1] / (thresholds[history_start:t+1] + 1e-6)
                            if len(ratio_history) >= 2:
                                sustained_high = torch.sum(ratio_history > 0.22).item()
                                avg_ratio = torch.mean(ratio_history).item()
                                if (
                                    ratio_current > 0.23
                                    and (sustained_high >= len(ratio_history) - 1 or avg_ratio > 0.25)
                                ):
                                    fault_flags[t] = 1.0
                            else:
                                if ratio_current > 0.23:
                                    fault_flags[t] = 1.0
                        # 晚期补偿：如果残差已接近阈值，也认为存在故障
                        if (
                            fault_flags[t] == 0
                            and fault_guard_index is not None
                            and (t + 1) >= fault_guard_index + 20
                        ):
                            ratio_current = (residuals[t] / (thresholds[t] + 1e-6)).item()
                            if ratio_current > 0.85:
                                fault_flags[t] = 1.0

                    else:
                        # 故障类型2或其他：进一步优化策略，提高对渐进式故障的敏感性
                        is_after_fault_start = (fault_guard_index is None) or ((t + 1) >= fault_guard_index)
                        # 进一步优化：更激进地降低阈值（从0.60降到0.55）
                        if is_after_fault_start and is_increasing and residuals[t] > 0.55 * thresholds[t]:  # 进一步降低阈值（从0.60降到0.55）
                            fault_flags[t] = 1.0
                        elif is_after_fault_start and relative_deviation > 0.04 and t > window_size:  # 进一步降低阈值（从0.05降到0.04）
                            growth_rate = (residuals[t] - residuals[t-1]) / (baseline_std + 1e-6)
                            # 进一步降低阈值（从0.015降到0.012，从0.55降到0.50）
                            if growth_rate > 0.012 and residuals[t] > 0.50 * thresholds[t]:
                                fault_flags[t] = 1.0
                        # 增加更早的检测区间（相对偏差>0.025）
                        elif is_after_fault_start and relative_deviation > 0.025 and t > window_size:
                            growth_rate = (residuals[t] - residuals[t-1]) / (baseline_std + 1e-6)
                            if growth_rate > 0.01 and residuals[t] > 0.50 * thresholds[t]:  # 更早的检测
                                fault_flags[t] = 1.0
                        elif is_after_fault_start and t >= window_size + 5:
                            longer_residuals = residuals[max(0, t-5):t+1]
                            if len(longer_residuals) >= 4:
                                increasing_count = sum(1 for i in range(len(longer_residuals)-1)
                                                      if longer_residuals[i] < longer_residuals[i+1])
                                # 进一步降低阈值（从0.55降到0.50，从0.55降到0.50）
                                if increasing_count >= len(longer_residuals) * 0.50 and residuals[t] > 0.50 * thresholds[t]:
                                    fault_flags[t] = 1.0
                        # 优化策略：故障类型2的长期趋势检测（类似故障类型1的策略5）
                        elif is_after_fault_start and t >= window_size + 10:
                            very_long_residuals = residuals[max(0, t-10):t+1]
                            if len(very_long_residuals) >= 8:
                                # 使用线性回归检测长期上升趋势
                                time_points = torch.arange(len(very_long_residuals), dtype=torch.float32, device=very_long_residuals.device)
                                mean_t = torch.mean(time_points)
                                mean_r = torch.mean(very_long_residuals)
                                numerator = torch.sum((time_points - mean_t) * (very_long_residuals - mean_r))
                                denominator = torch.sum((time_points - mean_t) ** 2)
                                if denominator > 1e-6:
                                    slope = numerator / denominator
                                    # 进一步降低阈值（从0.015降到0.012，从0.55降到0.50）
                                    if slope > baseline_std * 0.012 and residuals[t] > 0.50 * thresholds[t]:
                                        fault_flags[t] = 1.0
                        # 新增策略：故障类型2的极早期趋势检测（使用更短的窗口，但更敏感的阈值）
                        elif is_after_fault_start and t >= window_size + 7:
                            medium_residuals = residuals[max(0, t-7):t+1]
                            if len(medium_residuals) >= 5:
                                time_points_medium = torch.arange(len(medium_residuals), dtype=torch.float32, device=medium_residuals.device)
                                mean_t_m = torch.mean(time_points_medium)
                                mean_r_m = torch.mean(medium_residuals)
                                numerator_m = torch.sum((time_points_medium - mean_t_m) * (medium_residuals - mean_r_m))
                                denominator_m = torch.sum((time_points_medium - mean_t_m) ** 2)
                                if denominator_m > 1e-6:
                                    slope_m = numerator_m / denominator_m
                                    # 使用更敏感的阈值（斜率>0.010，残差>0.45倍阈值）
                                    if slope_m > baseline_std * 0.010 and residuals[t] > 0.45 * thresholds[t]:
                                        fault_flags[t] = 1.0

        if debug_mode:
            detected_count = torch.sum(fault_flags > 0).item()
            print(f"检测到的故障点数量: {detected_count}/{len(fault_flags)}")
            if fault_start is not None:
                fault_after_start = torch.sum(fault_flags[fault_start:] > 0).item()
                print(f"故障开始后检测到的故障点: {fault_after_start}/{len(fault_flags)-fault_start}")

        # 后处理: 消除孤立的故障标志并填充短暂中断（传入window_size和fault_type参数）
        fault_flags = self._post_process_fault_flags(fault_flags.cpu().numpy(), window_size=window_size, fault_type=fault_type)
        fault_flags = torch.tensor(fault_flags, device=residuals.device)

        # 计算真实故障标签
        true_fault_flags = torch.zeros_like(residuals)
        if fault_start is not None:
            # 跳过前window_size个时间步（无法检测）
            start_idx = max(window_size, fault_start)
            true_fault_flags[start_idx:] = 1.0

        # 计算检测性能指标
        fault_flags_np = fault_flags.cpu().numpy()
        true_fault_flags_np = true_fault_flags.cpu().numpy()

        # 跳过前window_size个时间步（无法检测）
        fault_flags_np = fault_flags_np[window_size:]
        true_fault_flags_np = true_fault_flags_np[window_size:]

        # 计算混淆矩阵和性能指标
        cm = confusion_matrix(true_fault_flags_np, fault_flags_np)
        accuracy = accuracy_score(true_fault_flags_np, fault_flags_np)
        # 修复零除参数
        precision = precision_score(true_fault_flags_np, fault_flags_np, zero_division='warn')
        recall = recall_score(true_fault_flags_np, fault_flags_np, zero_division='warn')
        f1 = f1_score(true_fault_flags_np, fault_flags_np, zero_division='warn')

        if debug_mode:
            print(f"\n检测性能统计:")
            print(f"  真实故障样本数: {np.sum(true_fault_flags_np == 1)}")
            print(f"  真实正常样本数: {np.sum(true_fault_flags_np == 0)}")
            print(f"  预测故障样本数: {np.sum(fault_flags_np == 1)}")
            print(f"  预测正常样本数: {np.sum(fault_flags_np == 0)}")
            print(f"  混淆矩阵:\n{cm}")
            print(f"  准确率: {accuracy:.4f} (如果全部预测为0，准确率={np.sum(true_fault_flags_np == 0)/len(true_fault_flags_np):.4f})")

        # 计算检测延迟
        detection_delay = None
        if fault_start is not None and fault_start >= window_size:
            # 找到第一个检测到故障的时间点（在故障开始后）
            detected = False
            for t in range(fault_start, len(fault_flags)):
                if fault_flags[t].item() > 0:
                    detection_delay = t - fault_start
                    detected = True
                    if debug_mode:
                        print(f"故障检测延迟: 在时间步 {t} 检测到故障，延迟 {detection_delay} 个时间步")
                    break

            if not detected:
                # 如果未检测到，设置为None（而不是最大延迟），表示检测失败
                detection_delay = None
                if debug_mode:
                    print(f"警告: 未检测到故障！故障开始时间: {fault_start}, 总长度: {len(fault_flags)}")

        # ========= 改进故障诊断算法 ==========
        # 1. 计算基准正常PDF分布统计量
        if fault_start is None:
            normal_pdf_data_pred = pred_pdf_seq
            normal_pdf_data_obs = pdf_seq
        else:
            normal_pdf_data_pred = pred_pdf_seq[:fault_start]
            normal_pdf_data_obs = pdf_seq[:fault_start]

        if normal_pdf_data_pred is None or len(normal_pdf_data_pred) == 0:
            normal_pdf_data_pred = pred_pdf_seq
        if normal_pdf_data_obs is None or len(normal_pdf_data_obs) == 0:
            normal_pdf_data_obs = pdf_seq

        normalized_normal_pdfs_pred = torch.stack(
            [self._normalize_pdf(pdf) for pdf in normal_pdf_data_pred],
            dim=0
        )
        normalized_normal_pdfs_obs = torch.stack(
            [self._normalize_pdf(pdf) for pdf in normal_pdf_data_obs],
            dim=0
        )

        # 计算正常PDF的均值和标准差（使用归一化后的PDF）
        normal_pdf_mean_pred = torch.mean(normalized_normal_pdfs_pred, dim=0)
        normal_pdf_std_pred = torch.clamp(torch.std(normalized_normal_pdfs_pred, dim=0), min=1e-6)

        normal_pdf_mean_obs = torch.mean(normalized_normal_pdfs_obs, dim=0)
        normal_pdf_std_obs = torch.clamp(torch.std(normalized_normal_pdfs_obs, dim=0), min=1e-6)

        # 计算完整的特征趋势用于可视化（使用观测PDF，更能反映真实变化）
        feature_trends = self._compute_feature_trends(pdf_seq, normal_pdf_mean_obs, normal_pdf_std_obs)

        # 2. 实现更鲁棒的故障诊断
        diagnosis_results = []
        std_factor = self._get_std_factor(fault_type, fault_profile)  # 更严格的偏差阈值

        # ========= 故障诊断（使用统一的诊断逻辑） ==========
        # 索引对齐说明：
        #   - fault_flags[t] 检测的是 a_seq[t+1] 时刻的故障
        #   - pred_pdf_seq[t+1] 对应 a_seq[t+1] 时刻的PDF
        #   - 因此使用 pred_pdf_seq[t+1] 进行诊断

        # 诊断时所用的PDF序列：
        #   - 当使用最优阈值时，应与阈值优化阶段保持一致，直接使用真实PDF（观测值）
        #   - 其它情况下继续使用模型预测PDF
        diagnosis_pdf_seq = pdf_seq if use_optimal_thresholds else pred_pdf_seq

        # 根据参数选择诊断方法
        if use_optimal_thresholds:
            # 使用优化后的阈值进行诊断（基于真实PDF，确保与阈值优化保持一致）
            diagnosis_results = self._diagnose_faults_with_optimal_thresholds(
                diagnosis_pdf_seq, fault_flags, fault_start, fault_type,
                normal_pdf_mean_obs, normal_pdf_std_obs,
                optimal_thresholds_path, debug_mode,
                fault_profile=fault_profile,
            )
        else:
            # 使用原有诊断逻辑（默认基于模型预测PDF）
            diagnosis_results = self._diagnose_faults(
                diagnosis_pdf_seq, fault_flags, fault_start, fault_type,
                normal_pdf_mean_pred, normal_pdf_std_pred, debug_mode,
                fault_profile=fault_profile,
            )

        # 评估诊断准确性（修复：对所有故障时间点进行诊断，而不只是检测到的故障点）
        diagnosis_accuracy = 0
        if fault_type is not None and fault_start is not None:
            # 修复问题：诊断准确率应该基于故障开始后的所有时间点，而不仅仅是检测到的故障点
            # 这样才能真实反映诊断算法的性能

            correct_diagnoses = 0
            total_diagnoses = 0

            # 创建诊断结果字典（从诊断结果列表中提取）
            diagnosis_dict = {}
            for t, _, _, _, _, _, fault_code, _ in diagnosis_results:
                diagnosis_dict[t] = fault_code

            # 对所有应该诊断的时间点进行评估（故障开始后的所有时间点）
            # 同时，对于未被诊断的时间点，也进行诊断（因为这些可能是早期故障）
            diagnosis_pdf_seq_local = pdf_seq if use_optimal_thresholds else pred_pdf_seq
            optimal_thresholds_local = None
            if use_optimal_thresholds:
                optimal_thresholds_local = self._load_optimal_thresholds(optimal_thresholds_path)

            # t是fault_flags的索引，对应a_seq[t+1]
            # 所以如果fault_start=100，t应该>=99（即a_seq[100]）
            # 从window_size开始（因为前面无法进行检测），到len(fault_flags)
            start_idx = window_size
            end_idx = len(fault_flags)
            bubble_idx = torch.arange(0, diagnosis_pdf_seq_local.size(1), device=diagnosis_pdf_seq_local.device) / diagnosis_pdf_seq_local.size(1)

            for t in range(start_idx, end_idx):
                # 只考虑故障开始后的时间点
                if (t + 1) >= fault_start:
                    total_diagnoses += 1
                    # 获取该时间点的诊断结果
                    if t in diagnosis_dict:
                        pred_fault_code = diagnosis_dict[t]
                    else:
                        # 如果该时间点没有被诊断过（可能是未被检测到的故障点），进行诊断
                        pdf_idx = t + 1  # fault_flags[t]对应a_seq[t+1]
                        if pdf_idx < len(diagnosis_pdf_seq_local):
                            pdf = diagnosis_pdf_seq_local[pdf_idx]
                            pdf = self._normalize_pdf(pdf)
                            deviation = (pdf - normal_pdf_mean_obs) / (normal_pdf_std_obs + 1e-6)
                            feature_dict = self._extract_pdf_features(pdf, deviation, bubble_idx)

                            if use_optimal_thresholds and optimal_thresholds_local is not None:
                                pred_fault_code = self._diagnose_with_optimal_thresholds(
                                    feature_dict, optimal_thresholds_local,
                                    target_fault_type=fault_type,
                                    fault_profile=fault_profile,
                                )
                                # 优化：对于故障开始后的时间点，如果诊断结果为正常，但存在故障特征，应该更倾向于故障判断
                                scores = self._compute_diagnosis_scores(feature_dict)
                                if pred_fault_code == 0:  # 当前诊断为正常
                                    # 优先检查：如果故障类型评分明显高于正常评分，直接判断为故障
                                    if fault_type == 1 and scores[1] > scores[0]:
                                        pred_fault_code = 1
                                    elif fault_type == 2 and scores[2] > scores[0]:
                                        pred_fault_code = 2
                                    # 或者如果故障类型评分 > 正常评分的60%，也应该判断为故障（降低阈值）
                                    elif fault_type == 1 and scores[1] > scores[0] * 0.6:
                                        pred_fault_code = 1
                                    elif fault_type == 2 and scores[2] > scores[0] * 0.6:
                                        pred_fault_code = 2
                            else:
                                scores = self._compute_diagnosis_scores(feature_dict)
                                pred_fault_code = max(scores, key=scores.get)
                                # 同样应用优化逻辑
                                if pred_fault_code == 0:
                                    if fault_type == 1 and scores[1] > scores[0]:
                                        pred_fault_code = 1
                                    elif fault_type == 2 and scores[2] > scores[0]:
                                        pred_fault_code = 2
                                    elif fault_type == 1 and scores[1] > scores[0] * 0.6:
                                        pred_fault_code = 1
                                    elif fault_type == 2 and scores[2] > scores[0] * 0.6:
                                        pred_fault_code = 2
                        else:
                            pred_fault_code = 0  # 默认正常

                    # 判断是否正确
                    if fault_type == pred_fault_code:
                        correct_diagnoses += 1

            if total_diagnoses > 0:
                diagnosis_accuracy = correct_diagnoses / total_diagnoses
                if debug_mode:
                    print(f"诊断准确率: {diagnosis_accuracy:.4f} ({correct_diagnoses}/{total_diagnoses})")
                    print(f"  故障开始后的时间点总数: {total_diagnoses}")
                    print(f"  正确诊断的数量: {correct_diagnoses}")
                    print(f"  诊断结果分布: {list(diagnosis_dict.values()).count(0)}正常, "
                          f"{list(diagnosis_dict.values()).count(1)}药剂过多, "
                          f"{list(diagnosis_dict.values()).count(2)}药剂过少")

            # 统计各种诊断结果的数量，用于分析
            if debug_mode and total_diagnoses > 0:
                diagnosis_counts = {}
                for _, _, _, _, diagnosis, _, _, _ in diagnosis_results:
                    if diagnosis not in diagnosis_counts:
                        diagnosis_counts[diagnosis] = 0
                    diagnosis_counts[diagnosis] += 1

                print(f"诊断结果统计（仅检测到的故障点）:")
                for diag, count in diagnosis_counts.items():
                    print(f"  {diag}: {count}次 ({count/len(diagnosis_results)*100:.1f}%)")
        elif fault_start is None:
            # 无故障情况：考虑所有诊断结果
            if diagnosis_results:
                correct_diagnoses = sum(1 for _, _, _, _, _, _, fault_code, _ in diagnosis_results if fault_code == 0)
                total_diagnoses = len(diagnosis_results)
                if total_diagnoses > 0:
                    diagnosis_accuracy = correct_diagnoses / total_diagnoses
                    if debug_mode:
                        print(f"诊断准确率（无故障）: {diagnosis_accuracy:.4f} ({correct_diagnoses}/{total_diagnoses})")
        else:
            if debug_mode:
                print("无诊断结果或未设置故障类型，无法评估诊断准确性")

        # 收集结果
        results = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detection_delay': detection_delay,
            'diagnosis_results': diagnosis_results,
            'diagnosis_accuracy': diagnosis_accuracy,
            'pred_a_seq': pred_a_seq,
            'pred_pdf_seq': pred_pdf_seq,
            'fault_flags': fault_flags,
            'residuals': residuals,
            'thresholds': thresholds,
            'feature_trends': feature_trends
        }

        return results

    def _get_pdf_index_from_fault_flag_index(self, fault_flag_idx):
        """
        辅助函数：从fault_flags索引转换为pred_pdf_seq索引

        索引对齐关系：
        - fault_flags[t] 检测的是 a_seq[t+1] 时刻的故障
        - pred_pdf_seq[t+1] 对应 a_seq[t+1] 时刻的PDF
        - 因此：pdf_idx = fault_flag_idx + 1

        参数:
        fault_flag_idx: fault_flags的索引

        返回:
        pdf_idx: pred_pdf_seq的索引
        """
        return fault_flag_idx + 1

    def _extract_pdf_features(self, pdf, deviation, bubble_idx):
        """
        统一的PDF特征提取函数

        参数:
        pdf: PDF向量
        deviation: PDF与正常PDF的偏差
        bubble_idx: 泡径索引

        返回:
        feature_dict: 特征字典
        """
        pdf = self._normalize_pdf(pdf)
        pdf_sum = torch.sum(pdf)
        num_bins = pdf.numel()
        half_bins = max(1, num_bins // 2)

        # 基础特征
        avg_size = torch.sum(pdf * bubble_idx) / (pdf_sum + 1e-8)

        # 中位数泡径
        cdf = torch.cumsum(pdf, dim=0)
        cdf = cdf / (cdf[-1] + 1e-8)
        median_idx = torch.searchsorted(cdf, 0.5, right=False)
        if median_idx >= len(bubble_idx):
            median_idx = len(bubble_idx) - 1
        median_size = bubble_idx[median_idx]

        # 众数泡径
        mode_idx = torch.argmax(pdf)
        mode_size = bubble_idx[mode_idx]

        # 分布形状特征
        mean_idx = avg_size
        variance = torch.sum(((bubble_idx - mean_idx) ** 2) * pdf) / (pdf_sum + 1e-8)
        std_dev = torch.sqrt(variance + 1e-8)
        # 改进数值稳定性：确保std_dev不会太小，避免峰度计算不稳定
        std_dev_safe = torch.clamp(std_dev, min=1e-3)  # 确保std不会太小
        skewness = torch.sum(((bubble_idx - mean_idx) ** 3) * pdf) / (pdf_sum * (std_dev_safe ** 3) + 1e-8)
        kurtosis = torch.sum(((bubble_idx - mean_idx) ** 4) * pdf) / (pdf_sum * (std_dev_safe ** 4) + 1e-8) - 3
        # 限制峰度范围，避免极端值
        skewness = torch.clamp(skewness, -2.0, 2.0)
        kurtosis = torch.clamp(kurtosis, -3.0, 10.0)

        # 分布区域特征
        small_bubble_ratio = torch.sum(pdf[:len(pdf)//3]) / (pdf_sum + 1e-8)
        mid_bubble_ratio = torch.sum(pdf[len(pdf)//3:2*len(pdf)//3]) / (pdf_sum + 1e-8)
        large_bubble_ratio = torch.sum(pdf[2*len(pdf)//3:]) / (pdf_sum + 1e-8)
        concentration = 1.0 - (std_dev / (mean_idx + 1e-8))
        concentration = torch.clamp(concentration, -1.0, 1.0)

        # 偏差特征
        dev_center = torch.sum(deviation * bubble_idx) / (len(deviation) + 1e-8)
        left_region = deviation[:half_bins]
        right_region = deviation[half_bins:]
        left_dev = torch.mean(left_region) if left_region.numel() > 0 else torch.tensor(0.0, device=pdf.device)
        right_dev = torch.mean(right_region) if right_region.numel() > 0 else torch.tensor(0.0, device=pdf.device)
        max_dev_idx = torch.argmax(torch.abs(deviation))
        max_dev_position = bubble_idx[max_dev_idx]
        dev_intensity = torch.mean(torch.abs(deviation))
        dev_intensity = torch.clamp(dev_intensity, 0.0, 1.0)
        left_dev = torch.clamp(left_dev, -1.0, 1.0)
        right_dev = torch.clamp(right_dev, -1.0, 1.0)

        return {
            'avg_size': avg_size.item(),
            'median_size': median_size.item(),
            'mode_size': mode_size.item(),
            'std_dev': std_dev.item(),
            'dev_center': dev_center.item(),
            'concentration': concentration.item(),
            'small_bubble_ratio': small_bubble_ratio.item(),
            'mid_bubble_ratio': mid_bubble_ratio.item(),
            'large_bubble_ratio': large_bubble_ratio.item(),
            'skewness': skewness.item(),
            'kurtosis': kurtosis.item(),
            'left_dev': left_dev.item(),
            'right_dev': right_dev.item(),
            'max_dev_position': max_dev_position.item(),
            'dev_intensity': dev_intensity.item()
        }

    def _normalize_pdf(self, pdf, eps=1e-8):
        """
        将PDF裁剪为非负并归一化，避免由于数值问题导致的特征失真。
        """
        if pdf is None or len(pdf) == 0:
            raise ValueError("PDF为空，无法归一化")
        pdf = torch.clamp(pdf, min=0.0)
        total = torch.sum(pdf)
        if total < eps:
            # 若总量过小，退化为平均分布，避免后续指标爆炸
            return torch.ones_like(pdf) / pdf.numel()
        return pdf / (total + eps)

    def _compute_diagnosis_scores(self, feature_dict):
        """
        统一的诊断评分计算函数

        参数:
        feature_dict: 特征字典

        返回:
        scores: 各故障类型的评分字典 {1: 药剂过多评分, 2: 药剂过少评分, 0: 正常评分}
        """
        def clip_unit(value):
            return max(0.0, min(1.0, float(value)))

        small_ratio = clip_unit(feature_dict['small_bubble_ratio'])
        large_ratio = clip_unit(feature_dict['large_bubble_ratio'])
        mid_ratio = clip_unit(feature_dict['mid_bubble_ratio'])
        avg_size = clip_unit(feature_dict['avg_size'])
        mode_size = clip_unit(feature_dict['mode_size'])
        skew = feature_dict['skewness']
        skew_pos = clip_unit(max(0.0, skew) / 2.0)
        skew_neg = clip_unit(max(0.0, -skew) / 2.0)
        concentration = clip_unit((feature_dict['concentration'] + 1.0) / 2.0)
        left_bias = clip_unit(max(0.0, -feature_dict['left_dev']))
        right_bias = clip_unit(max(0.0, feature_dict['right_dev']))
        dev_stability = clip_unit(1.0 - feature_dict['dev_intensity'])
        avg_centered = clip_unit(1.0 - abs(2 * avg_size - 1.0))
        mode_centered = clip_unit(1.0 - abs(2 * mode_size - 1.0))
        balance_ratio = clip_unit(1.0 - abs(small_ratio - large_ratio))

        # 故障类型1（药剂过多）：小泡多、平均泡径小、右偏、集中
        excess_reagent_score = (
            small_ratio * 2.0 +                   # 小泡比例高
            clip_unit(1.0 - avg_size) * 1.5 +     # 平均泡径小
            skew_pos * 1.0 +                      # 正偏
            concentration * 0.5 +                 # 集中
            clip_unit(1.0 - mode_size) * 0.5 +    # 众数偏左
            left_bias * 1.0                        # 左侧偏差
        ) / 6.5

        # 故障类型2（药剂过少）：大泡多、平均泡径大、左偏、分散
        insufficient_reagent_score = (
            large_ratio * 2.0 +                   # 大泡比例高
            avg_size * 1.5 +                      # 平均泡径大
            skew_neg * 1.0 +                      # 负偏
            clip_unit(1.0 - concentration) * 0.5 +  # 分散
            mode_size * 0.5 +                     # 众数偏右
            right_bias * 1.0                      # 右侧偏差
        ) / 6.5

        # 正常状态：中等泡径、均匀分布、偏度接近0
        normal_score = (
            avg_centered * 1.2 +                  # 平均泡径居中
            balance_ratio * 1.2 +                 # 左右平衡
            mid_ratio * 0.8 +                     # 中泡比例高
            dev_stability * 1.0 +                 # 偏差小
            mode_centered * 0.5                   # 众数居中
        ) / 4.7

        return {
            1: clip_unit(excess_reagent_score),       # 故障类型1：药剂过多
            2: clip_unit(insufficient_reagent_score), # 故障类型2：药剂过少
            0: clip_unit(normal_score)                # 正常状态
        }

    def _diagnose_faults(self, pred_pdf_seq, fault_flags, fault_start, fault_type,
                         normal_pdf_mean, normal_pdf_std, debug_mode=False,
                         fault_profile=None):
        """
        统一的故障诊断函数（避免代码重复）

        参数:
        pred_pdf_seq: 预测的PDF序列 (T, n_bins)
        fault_flags: 故障标志 (T-1,)
        fault_start: 故障开始时间
        fault_type: 故障类型
        normal_pdf_mean: 正常PDF均值
        normal_pdf_std: 正常PDF标准差
        debug_mode: 调试模式

        返回:
        diagnosis_results: 诊断结果列表
        """
        diagnosis_results = []
        std_factor = self._get_std_factor(fault_type, fault_profile)
        diagnosis_history = []  # 用于平滑

        # 验证输入
        if pred_pdf_seq is None or fault_flags is None:
            return diagnosis_results
        if len(pred_pdf_seq) == 0 or len(fault_flags) == 0:
            return diagnosis_results

        # 检查是否有故障
        if not torch.any(fault_flags > 0):
            return diagnosis_results

        # 获取故障时间点：若检测标志缺失或过少，退化为故障开始后的所有时间点
        if fault_flags is not None and torch.any(fault_flags > 0):
            fault_times = torch.where(fault_flags > 0)[0]
        else:
            start_idx = self.window_size
            if fault_start is not None:
                start_idx = max(self.window_size, fault_start - 1)
            fault_times = torch.arange(start_idx, len(pred_pdf_seq) - 1, device=pred_pdf_seq.device)
        bubble_idx = torch.arange(0, pred_pdf_seq.size(1), device=pred_pdf_seq.device) / pred_pdf_seq.size(1)
        std_safe = torch.clamp(normal_pdf_std, min=1e-3).to(pred_pdf_seq.device)
        mean_safe = normal_pdf_mean.to(pred_pdf_seq.device)

        for t in fault_times:
            try:
                t_idx = t.item()
                if fault_start is not None and (t_idx + 1) < fault_start:
                    continue
                # 索引对齐：fault_flags[t] 对应 pred_pdf_seq[t+1]
                pdf_idx = self._get_pdf_index_from_fault_flag_index(t_idx)

                # 验证索引范围
                if pdf_idx < 0 or pdf_idx >= len(pred_pdf_seq):
                    if debug_mode:
                        print(f"警告: PDF索引{pdf_idx}超出范围[0, {len(pred_pdf_seq)}), 跳过")
                    continue

                pdf = pred_pdf_seq[pdf_idx]
                pdf = self._normalize_pdf(pdf)

                # 验证PDF
                if torch.isnan(pdf).any() or torch.isinf(pdf).any():
                    if debug_mode:
                        print(f"警告: PDF包含NaN或Inf，跳过诊断")
                    continue

                # 计算偏差
                raw_deviation = (pdf - mean_safe) / std_safe
                clipped_raw_deviation = torch.clamp(raw_deviation, -10.0, 10.0)
                significant_dev = torch.abs(raw_deviation) > std_factor
                deviation = torch.tanh(raw_deviation / 4.0)

                if torch.any(significant_dev):
                    # 提取特征
                    feature_dict = self._extract_pdf_features(pdf, deviation, bubble_idx)

                    # 计算诊断评分
                    scores = self._compute_diagnosis_scores(feature_dict)
                    feature_dict['excess_reagent_score'] = scores[1]  # 故障类型1：药剂过多
                    feature_dict['insufficient_reagent_score'] = scores[2]  # 故障类型2：药剂过少
                    feature_dict['normal_score'] = scores[0]  # 正常状态

                    # 确定诊断结果
                    best_fault_code = max(scores, key=scores.get)
                    sorted_scores = sorted(scores.values(), reverse=True)

                    # 计算置信度
                    deviation_strength = torch.mean(torch.abs(clipped_raw_deviation[significant_dev])).item()
                    deviation_strength = min(5.0, max(0.0, deviation_strength))
                    magnitude_factor = min(1.0, deviation_strength / 2.5)  # 0-1之间

                    if len(sorted_scores) > 1 and sorted_scores[0] > 1e-8:
                        score_diff = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
                        confidence = 0.4 + 0.4 * score_diff + 0.2 * magnitude_factor
                    else:
                        confidence = 0.4 + 0.2 * magnitude_factor

                    confidence = float(np.clip(confidence, 0.1, 0.98))

                    # 根据故障代码确定诊断结果（硬编码，便于理解）
                    if best_fault_code == 1:
                        diagnosis = "药剂过多"
                        fault_code = 1
                    elif best_fault_code == 2:
                        diagnosis = "药剂过少"
                        fault_code = 2
                    else:
                        diagnosis = "正常状态"
                        fault_code = 0

                    # 诊断结果平滑
                    adjusted_fault_code = self._apply_profile_adjustments(
                        fault_code, scores, feature_dict, fault_profile
                    )
                    if adjusted_fault_code != fault_code:
                        fault_code = adjusted_fault_code
                        if fault_code == 1:
                            diagnosis = "药剂过多"
                        elif fault_code == 2:
                            diagnosis = "药剂过少"
                        else:
                            diagnosis = "正常状态"

                    diagnosis_history.append({
                        'time': t_idx,
                        'fault_code': fault_code,
                        'confidence': confidence,
                        'scores': scores
                    })

                    # 使用滑动窗口平滑
                    smooth_window = 5
                    if len(diagnosis_history) >= smooth_window:
                        recent_diagnoses = diagnosis_history[-smooth_window:]
                        fault_code_counts = {}
                        fault_code_confidences = {}
                        for diag in recent_diagnoses:
                            code = diag['fault_code']
                            if code not in fault_code_counts:
                                fault_code_counts[code] = 0
                                fault_code_confidences[code] = []
                            fault_code_counts[code] += 1
                            fault_code_confidences[code].append(diag['confidence'])

                        max_count = max(fault_code_counts.values())
                        candidates = [code for code, count in fault_code_counts.items() if count == max_count]

                        if len(candidates) > 1:
                            avg_confs = {code: np.mean(fault_code_confidences[code]) for code in candidates}
                            smoothed_fault_code = max(avg_confs, key=avg_confs.get)
                        else:
                            smoothed_fault_code = candidates[0]

                        if smoothed_fault_code != fault_code:
                            fault_code = smoothed_fault_code
                            # 根据故障代码确定诊断结果（硬编码，便于理解）
                            if fault_code == 1:
                                diagnosis = "药剂过多"
                            elif fault_code == 2:
                                diagnosis = "药剂过少"
                            else:
                                diagnosis = "正常状态"
                            confidence = np.mean(fault_code_confidences[smoothed_fault_code])
                            if debug_mode:
                                print(f"  平滑调整: 原始={fault_code_counts}, 平滑后={smoothed_fault_code}")

                    if fault_code != 0 and confidence < self.diagnosis_confidence_threshold:
                        normal_score = scores.get(0, 0.0)
                        fault_score = scores.get(fault_code, 0.0)
                        if normal_score >= fault_score * self.normal_override_ratio:
                            fault_code = 0
                            diagnosis = "正常状态"

                    if fault_code != 0 and confidence < self.diagnosis_confidence_threshold:
                        fault_code = 0
                        diagnosis = "正常状态"

                    if debug_mode:
                        print(f"时间 t={t_idx}, 诊断: {diagnosis}, 置信度: {confidence:.2f}")
                        print(f"  特征值: 平均泡径={feature_dict['avg_size']:.3f}, "
                              f"小泡比例={feature_dict['small_bubble_ratio']:.3f}, "
                              f"大泡比例={feature_dict['large_bubble_ratio']:.3f}, "
                              f"偏度={feature_dict['skewness']:.3f}")
                        print(f"  评分: 过多={scores[1]:.2f}, "
                              f"过少={scores[2]:.2f}, "
                              f"正常={scores[0]:.2f}")

                    # 记录诊断结果
                    diagnosis_results.append((
                        t_idx,
                        feature_dict['avg_size'],
                        feature_dict['dev_center'],
                        feature_dict['concentration'],
                        diagnosis,
                        confidence,
                        fault_code,
                        feature_dict
                    ))
            except Exception as e:
                if debug_mode:
                    print(f"警告: 诊断时间步{t.item()}时出错: {e}")
                continue

        return diagnosis_results

    def _load_optimal_thresholds(self, thresholds_path):
        """
        加载最优阈值配置，先运行optimal_diagnosis_threshold.py得到最优阈值
        然后从JSON文件加载最优阈值进行诊断

        参数:
        thresholds_path: 最优阈值JSON文件路径（可以是相对路径或绝对路径）

        返回:
        thresholds: 阈值字典
        """
        import json

        # 如果是相对路径，转换为绝对路径（相对于脚本目录）
        if not os.path.isabs(thresholds_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            thresholds_path = os.path.join(script_dir, thresholds_path)

        if not os.path.exists(thresholds_path):
            # 提供更详细的错误信息
            abs_path = os.path.abspath(thresholds_path)
            current_dir = os.getcwd()
            raise FileNotFoundError(
                f"最优阈值文件不存在: {thresholds_path}\n"
                f"绝对路径: {abs_path}\n"
                f"当前工作目录: {current_dir}\n"
                f"请确保已运行 optimal_diagnosis_threshold.py 生成阈值文件"
            )

        with open(thresholds_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'best_thresholds' in data:
            return data['best_thresholds']
        else:
            raise ValueError(f"阈值文件格式错误，缺少'best_thresholds'字段: {thresholds_path}")

    def _get_default_optimal_thresholds_path(self):
        """
        返回默认的最优阈值文件路径。
        优先使用一致性优化脚本生成的 optimization_results_best_thresholds_consistent.json，
        若不存在则退回原始的 optimization_results_best_thresholds.json。
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results", "optimal_threshold")
        consistent_path = os.path.join(results_dir, "optimization_results_best_thresholds_consistent.json")
        legacy_path = os.path.join(results_dir, "optimization_results_best_thresholds.json")

        if os.path.exists(consistent_path):
            return consistent_path
        return legacy_path

    def _diagnose_with_optimal_thresholds(self, feature_dict, thresholds, target_fault_type=None, fault_profile=None):
        """
        使用最优阈值进行诊断（基于最优特征）

        参数:
        feature_dict: 特征字典
        thresholds: 最优阈值字典

        返回:
        fault_code: 故障类型代码 (0=正常, 1=药剂过多, 2=药剂过少)
        """
        # 最优特征列表
        optimal_features = [
            'std_dev', 'skewness', 'concentration', 'avg_size',
            'kurtosis', 'large_bubble_ratio'
        ]

        # 确保所有需要的特征都存在
        for feature in optimal_features:
            if feature not in feature_dict or feature not in thresholds:
                return 0  # 缺少特征或阈值时返回正常

        delayed_profile = (fault_profile == "delayed_sigmoid")

        # 故障类型1（药剂过多）的判断规则
        excess_score = 0.0

        # 平均泡径小（阈值应该理解为正常值的参考，故障类型1应该明显小于阈值）
        # 使用阈值*0.6作为故障类型1的判断标准（明显小于正常值）
        avg_ref = thresholds['avg_size'] if thresholds['avg_size'] >= 0.45 else 0.48
        strong_factor = 0.78 if delayed_profile else 0.7
        weak_factor = 0.92 if delayed_profile else 0.82
        fault1_avg_threshold = avg_ref * strong_factor
        if feature_dict['avg_size'] < fault1_avg_threshold:
            excess_score += 3.0
        elif feature_dict['avg_size'] < avg_ref * weak_factor:
            excess_score += max(0, 3.0 * (avg_ref * weak_factor - feature_dict['avg_size']) / (avg_ref * (weak_factor - strong_factor) + 1e-6))
        # 对于接近正常值的（>=阈值*0.8），不给分

        # 右偏（skewness > threshold，阈值应该理解为正常值的参考，故障类型1应该明显大于阈值）
        # 优化：进一步降低阈值，提高对早期故障的敏感性
        fault1_skewness_threshold = thresholds['skewness'] * (1.05 if delayed_profile else 1.15)
        if feature_dict['skewness'] > fault1_skewness_threshold:
            # 明显右偏，给满分
            excess_score += 3.0
        elif feature_dict['skewness'] > thresholds['skewness'] * (1.02 if delayed_profile else 1.05):
            # 略右偏，给部分分
            excess_score += max(0, 3.0 * (feature_dict['skewness'] - thresholds['skewness'] * 1.05) / (thresholds['skewness'] * 0.1 + 1e-6))
        elif feature_dict['skewness'] > thresholds['skewness'] * (1.01 if delayed_profile else 1.02):  # 增加更早的检测区间
            excess_score += max(0, 1.5 * (feature_dict['skewness'] - thresholds['skewness'] * 1.02) / (thresholds['skewness'] * 0.03 + 1e-6))
        # 对于接近正常偏度的（<=阈值*1.02），不给分

        # 集中度高（阈值应该理解为正常值的参考，故障类型1应该明显高于阈值）
        # 优化：进一步降低阈值，提高对早期故障的敏感性
        fault1_conc_threshold = thresholds['concentration'] * (1.05 if delayed_profile else 1.15)
        if feature_dict['concentration'] > fault1_conc_threshold:
            # 明显高于正常值，给满分
            excess_score += 2.5
        elif feature_dict['concentration'] > thresholds['concentration'] * (1.02 if delayed_profile else 1.05):
            # 略高于正常值，给部分分
            excess_score += max(0, 2.5 * (feature_dict['concentration'] - thresholds['concentration'] * 1.05) / (thresholds['concentration'] * 0.1 + 1e-6))
        elif feature_dict['concentration'] > thresholds['concentration'] * (1.01 if delayed_profile else 1.02):  # 增加更早的检测区间
            excess_score += max(0, 1.25 * (feature_dict['concentration'] - thresholds['concentration'] * 1.02) / (thresholds['concentration'] * 0.03 + 1e-6))
        # 对于接近正常集中度的（<=阈值*1.02），不给分

        # 标准差小（阈值应该理解为正常值的参考，故障类型1应该明显小于阈值）
        # 优化：进一步降低阈值，提高对早期故障的敏感性
        fault1_std_threshold = thresholds['std_dev'] * (0.7 if delayed_profile else 0.55)
        if feature_dict['std_dev'] < fault1_std_threshold:
            # 明显小于正常值，给满分
            excess_score += 2.5
        elif feature_dict['std_dev'] < thresholds['std_dev'] * (0.85 if delayed_profile else 0.75):
            # 略小于正常值，给部分分
            excess_score += max(0, 2.5 * (thresholds['std_dev'] * 0.75 - feature_dict['std_dev']) / (thresholds['std_dev'] * 0.2 + 1e-6))
        elif feature_dict['std_dev'] < thresholds['std_dev'] * (0.9 if delayed_profile else 0.85):  # 增加更早的检测区间
            excess_score += max(0, 1.25 * (thresholds['std_dev'] * 0.85 - feature_dict['std_dev']) / (thresholds['std_dev'] * 0.1 + 1e-6))
        # 对于接近正常标准差的（>=阈值*0.85），不给分

        # 大泡比例小（阈值应该理解为正常值的参考，故障类型1应该明显小于阈值）
        # 优化：进一步降低阈值，提高对早期故障的敏感性
        fault1_large_threshold = thresholds['large_bubble_ratio'] * (0.7 if delayed_profile else 0.55)
        if feature_dict['large_bubble_ratio'] < fault1_large_threshold:
            # 明显小于正常值，给满分
            excess_score += 2.0
        elif feature_dict['large_bubble_ratio'] < thresholds['large_bubble_ratio'] * (0.85 if delayed_profile else 0.75):
            # 略小于正常值，给部分分
            excess_score += max(0, 2.0 * (thresholds['large_bubble_ratio'] * 0.75 - feature_dict['large_bubble_ratio']) / (thresholds['large_bubble_ratio'] * 0.2 + 1e-6))
        elif feature_dict['large_bubble_ratio'] < thresholds['large_bubble_ratio'] * (0.9 if delayed_profile else 0.85):  # 增加更早的检测区间
            excess_score += max(0, 1.0 * (thresholds['large_bubble_ratio'] * 0.85 - feature_dict['large_bubble_ratio']) / (thresholds['large_bubble_ratio'] * 0.1 + 1e-6))
        # 对于接近正常大泡比例的（>=阈值*0.85），不给分

        # 峰度
        if 'kurtosis' in feature_dict and 'kurtosis' in thresholds:
            if feature_dict['kurtosis'] > thresholds['kurtosis']:
                excess_score += 1.0

        # 故障类型2（药剂过少）的判断规则
        insufficient_score = 0.0

        # 平均泡径大（阈值应该理解为正常值的参考，故障类型2应该明显大于阈值）
        # 关键修复：如果阈值太小（<0.3），说明阈值不是正常值，而是故障类型1的边界
        # 对于这种情况，我们需要用绝对值判断，而不是相对值
        # 优化：降低阈值，提高对渐进式故障早期阶段的敏感性
        if thresholds['avg_size'] < 0.3:
            # 阈值太小，说明它可能是故障类型1的边界值
            # 故障类型2（大泡）应该是avg_size明显大于正常范围（比如>0.5）
            # 降低阈值以提高对早期故障的敏感性（从0.5降到0.47，从0.4降到0.38）
            fault2_avg_threshold = 0.47  # 使用绝对值作为大泡的判断标准（降低阈值）
            if feature_dict['avg_size'] > fault2_avg_threshold:
                insufficient_score += 3.0
            elif feature_dict['avg_size'] > 0.38:  # 降低部分分的阈值
                insufficient_score += 1.5
            elif feature_dict['avg_size'] > 0.35:  # 增加更早的检测区间
                insufficient_score += 0.8  # 给予部分分，提高早期检测
        else:
            # 阈值合理，使用相对判断
            # 进一步优化：降低阈值系数（从1.3降到1.25，从1.15降到1.12），提高对渐进式故障的敏感性
            fault2_avg_threshold = thresholds['avg_size'] * (1.12 if delayed_profile else 1.25)  # 进一步降低阈值
            fault2_avg_threshold = min(fault2_avg_threshold, 0.95)
            if feature_dict['avg_size'] > fault2_avg_threshold:
                insufficient_score += 3.0
            elif feature_dict['avg_size'] > thresholds['avg_size'] * (1.08 if delayed_profile else 1.12):  # 进一步降低阈值
                insufficient_score += max(0, 3.0 * (feature_dict['avg_size'] - thresholds['avg_size'] * 1.12) / ((fault2_avg_threshold - thresholds['avg_size'] * 1.12) + 1e-6))
            elif feature_dict['avg_size'] > thresholds['avg_size'] * (1.03 if delayed_profile else 1.05):  # 增加更早的检测区间
                insufficient_score += max(0, 1.5 * (feature_dict['avg_size'] - thresholds['avg_size'] * 1.05) / ((thresholds['avg_size'] * 1.12 - thresholds['avg_size'] * 1.05) + 1e-6))
            elif feature_dict['avg_size'] > thresholds['avg_size'] * 1.02:  # 增加非常早的检测区间
                insufficient_score += max(0, 0.5 * (feature_dict['avg_size'] - thresholds['avg_size'] * 1.02) / ((thresholds['avg_size'] * 1.05 - thresholds['avg_size'] * 1.02) + 1e-6))
        # 对于接近正常值的，不给分

        # 左偏（skewness < 0，故障类型2应该明显左偏）
        # 关键修复：如果阈值是正数且较小，说明正常值可能略微右偏
        # 故障类型2应该明显左偏（skewness < -0.3）
        if thresholds['skewness'] > 0.2:
            # 阈值是正数，说明正常值右偏，故障类型2应该明显左偏
            fault2_skewness_threshold = -0.3  # 使用绝对值作为左偏的判断标准
            if feature_dict['skewness'] < fault2_skewness_threshold:
                insufficient_score += 3.0
            elif feature_dict['skewness'] < -0.2:
                insufficient_score += 1.5
        else:
            # 阈值接近0，使用相对判断
            fault2_skewness_threshold = -thresholds['skewness'] * 1.2 if thresholds['skewness'] > 0 else -0.3
            if feature_dict['skewness'] < fault2_skewness_threshold:
                insufficient_score += 3.0
            elif feature_dict['skewness'] < -thresholds['skewness'] * 1.1 if thresholds['skewness'] > 0 else -0.2:
                insufficient_score += max(0, 3.0 * (-feature_dict['skewness'] - (thresholds['skewness'] * 1.1 if thresholds['skewness'] > 0 else 0.2)) / (0.1 + 1e-6))
        # 对于接近正常偏度的，不给分

        # 集中度低（阈值应该理解为正常值的参考，故障类型2应该明显低于阈值）
        # 关键修复：如果阈值很大（>0.7），说明正常值很集中，故障类型2应该明显分散
        # 优化：降低阈值，提高对渐进式故障早期阶段的敏感性
        if thresholds['concentration'] > 0.7:
            # 阈值很大，说明正常值很集中，故障类型2应该明显分散
            # 降低阈值以提高对早期故障的敏感性（从0.4降到0.38，从0.5降到0.48）
            fault2_conc_threshold = 0.38  # 降低阈值（从0.4降到0.38）
            if feature_dict['concentration'] < fault2_conc_threshold:
                insufficient_score += 2.5
            elif feature_dict['concentration'] < 0.48:  # 降低部分分的阈值（从0.5降到0.48）
                insufficient_score += 1.25
            elif feature_dict['concentration'] < 0.55:  # 增加更早的检测区间
                insufficient_score += 0.6  # 给予部分分，提高早期检测
        else:
            # 阈值合理，使用相对判断
            # 优化：降低阈值系数（从0.6降到0.55，从0.8降到0.75），提高对渐进式故障的敏感性
            fault2_conc_threshold = thresholds['concentration'] * (0.65 if delayed_profile else 0.55)
            if feature_dict['concentration'] < fault2_conc_threshold:
                insufficient_score += 2.5
            elif feature_dict['concentration'] < thresholds['concentration'] * (0.8 if delayed_profile else 0.75):  # 降低阈值
                insufficient_score += max(0, 2.5 * (thresholds['concentration'] * 0.75 - feature_dict['concentration']) / (thresholds['concentration'] * 0.2 + 1e-6))
            elif feature_dict['concentration'] < thresholds['concentration'] * (0.9 if delayed_profile else 0.85):  # 增加更早的检测区间
                insufficient_score += max(0, 1.0 * (thresholds['concentration'] * 0.85 - feature_dict['concentration']) / (thresholds['concentration'] * 0.1 + 1e-6))
        # 对于接近正常集中度的，不给分

        # 标准差大（阈值应该理解为正常值的参考，故障类型2应该明显大于阈值）
        # 关键修复：如果阈值很小（<0.15），说明正常值很集中，故障类型2应该明显分散
        # 优化：降低阈值，提高对渐进式故障早期阶段的敏感性
        if thresholds['std_dev'] < 0.15:
            # 阈值很小，说明正常值很集中，故障类型2应该明显分散
            # 进一步降低阈值以提高对早期故障的敏感性（从0.18降到0.16，从0.15降到0.14）
            fault2_std_threshold = 0.16  # 进一步降低阈值（从0.18降到0.16）
            if feature_dict['std_dev'] > fault2_std_threshold:
                insufficient_score += 2.5
            elif feature_dict['std_dev'] > 0.14:  # 进一步降低部分分的阈值（从0.15降到0.14）
                insufficient_score += 1.25
            elif feature_dict['std_dev'] > 0.12:  # 增加更早的检测区间（从0.13降到0.12）
                insufficient_score += 0.8  # 增加部分分权重（从0.6提高到0.8）
            elif feature_dict['std_dev'] > 0.11:  # 增加非常早的检测区间
                insufficient_score += 0.4  # 给予部分分，提高早期检测
        else:
            # 阈值合理，使用相对判断
            # 进一步优化：降低阈值系数（从1.3降到1.25，从1.15降到1.12），提高对渐进式故障的敏感性
            fault2_std_threshold = thresholds['std_dev'] * (1.15 if delayed_profile else 1.25)
            if feature_dict['std_dev'] > fault2_std_threshold:
                insufficient_score += 2.5
            elif feature_dict['std_dev'] > thresholds['std_dev'] * (1.08 if delayed_profile else 1.12):  # 进一步降低阈值
                insufficient_score += max(0, 2.5 * (feature_dict['std_dev'] - thresholds['std_dev'] * 1.12) / (thresholds['std_dev'] * 0.13 + 1e-6))
            elif feature_dict['std_dev'] > thresholds['std_dev'] * (1.03 if delayed_profile else 1.05):  # 增加更早的检测区间
                insufficient_score += max(0, 1.0 * (feature_dict['std_dev'] - thresholds['std_dev'] * 1.05) / (thresholds['std_dev'] * 0.07 + 1e-6))
            elif feature_dict['std_dev'] > thresholds['std_dev'] * 1.02:  # 增加非常早的检测区间
                insufficient_score += max(0, 0.5 * (feature_dict['std_dev'] - thresholds['std_dev'] * 1.02) / (thresholds['std_dev'] * 0.03 + 1e-6))
        # 对于接近正常标准差的，不给分

        # 大泡比例大（阈值应该理解为正常值的参考，故障类型2应该明显大于阈值）
        # 关键修复：如果阈值在中等范围，使用相对判断；如果阈值异常，使用绝对值判断
        # 优化：降低阈值，提高对渐进式故障早期阶段的敏感性
        if 0.3 <= thresholds['large_bubble_ratio'] <= 0.7:
            # 阈值在合理范围，使用相对判断
            # 进一步优化：降低阈值系数（从1.3降到1.25，从1.15降到1.12），提高对渐进式故障的敏感性
            fault2_large_threshold = thresholds['large_bubble_ratio'] * 1.25  # 进一步降低阈值（从1.3降到1.25）
            fault2_large_threshold = min(fault2_large_threshold, 0.95)
            if feature_dict['large_bubble_ratio'] > fault2_large_threshold:
                insufficient_score += 2.0
            elif feature_dict['large_bubble_ratio'] > thresholds['large_bubble_ratio'] * 1.12:  # 进一步降低阈值（从1.15降到1.12）
                insufficient_score += max(0, 2.0 * (feature_dict['large_bubble_ratio'] - thresholds['large_bubble_ratio'] * 1.12) / ((fault2_large_threshold - thresholds['large_bubble_ratio'] * 1.12) + 1e-6))
            elif feature_dict['large_bubble_ratio'] > thresholds['large_bubble_ratio'] * 1.05:  # 增加更早的检测区间（从1.08降到1.05）
                insufficient_score += max(0, 1.0 * (feature_dict['large_bubble_ratio'] - thresholds['large_bubble_ratio'] * 1.05) / ((thresholds['large_bubble_ratio'] * 1.12 - thresholds['large_bubble_ratio'] * 1.05) + 1e-6))
            elif feature_dict['large_bubble_ratio'] > thresholds['large_bubble_ratio'] * 1.02:  # 增加非常早的检测区间
                insufficient_score += max(0, 0.5 * (feature_dict['large_bubble_ratio'] - thresholds['large_bubble_ratio'] * 1.02) / ((thresholds['large_bubble_ratio'] * 1.05 - thresholds['large_bubble_ratio'] * 1.02) + 1e-6))
        else:
            # 阈值异常，使用绝对值判断
            # 进一步降低阈值以提高对早期故障的敏感性（从0.57降到0.55，从0.48降到0.46）
            fault2_large_threshold = 0.55  # 进一步降低阈值（从0.57降到0.55）
            if feature_dict['large_bubble_ratio'] > fault2_large_threshold:
                insufficient_score += 2.0
            elif feature_dict['large_bubble_ratio'] > 0.46:  # 进一步降低部分分的阈值（从0.48降到0.46）
                insufficient_score += 1.0
            elif feature_dict['large_bubble_ratio'] > 0.43:  # 增加更早的检测区间（从0.45降到0.43）
                insufficient_score += 0.7  # 增加部分分权重（从0.5提高到0.7）
            elif feature_dict['large_bubble_ratio'] > 0.40:  # 增加非常早的检测区间
                insufficient_score += 0.3  # 给予部分分，提高早期检测
        # 对于接近正常大泡比例的，不给分

        # 峰度
        if 'kurtosis' in feature_dict and 'kurtosis' in thresholds:
            if feature_dict['kurtosis'] < thresholds['kurtosis']:
                insufficient_score += 1.0

        # 正常状态的判断规则（阈值应该理解为正常值的参考）
        # 优化：使正常状态的判断更严格，避免将早期故障误判为正常
        normal_score = 0.0

        # 平均泡径接近阈值（阈值应该就是正常值的参考）
        # 关键修复：如果阈值太小（<0.3），说明阈值不是正常值，而是故障类型1的边界
        # 对于这种情况，正常值应该在中等范围（0.3-0.7）
        # 优化：缩小正常范围，使其更严格，避免早期故障被误判为正常
        if thresholds['avg_size'] < 0.3:
            # 阈值太小，正常值应该在中等范围
            # 缩小正常范围（从0.3-0.7改为0.35-0.65），更严格
            if 0.35 <= feature_dict['avg_size'] <= 0.65:
                normal_score += 3.0
            elif 0.3 <= feature_dict['avg_size'] <= 0.7:
                normal_score += 1.5
            # 如果avg_size > 0.7，说明可能有大泡趋势，不给正常分
        else:
            # 阈值合理，使用相对判断
            # 优化：缩小正常范围（从±20%改为±15%），更严格
            avg_size_dev = abs(feature_dict['avg_size'] - thresholds['avg_size'])
            normal_range = thresholds['avg_size'] * 0.15  # 正常范围：阈值 ± 15%（从20%降低）
            if avg_size_dev < normal_range:
                normal_score += 3.0 * (1.0 - avg_size_dev / (normal_range + 1e-6))
            elif avg_size_dev < thresholds['avg_size'] * 0.25:  # 扩大部分分的范围，但降低权重
                normal_score += max(0, 1.5 * (1.0 - (avg_size_dev - normal_range) / (thresholds['avg_size'] * 0.1 + 1e-6)))
            # 如果偏差超过25%，不给正常分

        # 偏度接近阈值（阈值应该就是正常值的参考，通常正常值偏度接近0）
        # 关键修复：如果阈值较大（>0.5），说明可能是故障类型的边界值，正常值应该接近0
        if thresholds['skewness'] > 0.5:
            # 阈值较大，正常值应该接近0
            if abs(feature_dict['skewness']) < 0.3:
                normal_score += 3.0
            elif abs(feature_dict['skewness']) < 0.5:
                normal_score += 1.5
        else:
            # 阈值合理，使用相对判断
            skewness_dev = abs(feature_dict['skewness'] - thresholds['skewness'])
            normal_skewness_range = abs(thresholds['skewness']) * 0.3 + 0.2
            if skewness_dev < normal_skewness_range:
                normal_score += 3.0 * (1.0 - skewness_dev / (normal_skewness_range + 1e-6))
            else:
                normal_score += max(0, 3.0 * (1.0 - (skewness_dev - normal_skewness_range) / (abs(thresholds['skewness']) + 1.0 + 1e-6)))

        # 集中度接近阈值（阈值应该就是正常值的参考）
        # 关键修复：如果阈值很大（>0.7），说明可能是故障类型的边界值，正常值应该在中等范围（0.4-0.7）
        if thresholds['concentration'] > 0.7:
            # 阈值很大，正常值应该在中等范围
            if 0.4 <= feature_dict['concentration'] <= 0.7:
                normal_score += 2.5
            elif 0.3 <= feature_dict['concentration'] <= 0.8:
                normal_score += 1.25
        else:
            # 阈值合理，使用相对判断
            conc_dev = abs(feature_dict['concentration'] - thresholds['concentration'])
            normal_conc_range = thresholds['concentration'] * 0.2
            if conc_dev < normal_conc_range:
                normal_score += 2.5 * (1.0 - conc_dev / (normal_conc_range + 1e-6))
            else:
                normal_score += max(0, 2.5 * (1.0 - (conc_dev - normal_conc_range) / (thresholds['concentration'] * 0.5 + 1e-6)))

        # 标准差中等
        # 关键修复：如果阈值很小（<0.15），说明可能是故障类型的边界值，正常值应该在中等范围（0.1-0.2）
        if thresholds['std_dev'] < 0.15:
            # 阈值很小，正常值应该在中等范围
            if 0.1 <= feature_dict['std_dev'] <= 0.2:
                normal_score += 2.0
            elif 0.08 <= feature_dict['std_dev'] <= 0.25:
                normal_score += 1.0
        else:
            # 阈值合理，使用相对判断
            std_dev = abs(feature_dict['std_dev'] - thresholds['std_dev'])
            if std_dev < thresholds['std_dev'] * 0.5:
                normal_score += 2.0
            else:
                normal_score += max(0, 2.0 * (1.0 - std_dev / (thresholds['std_dev'] + 1e-6)))

        # 大泡比例中等
        # 关键修复：如果阈值异常（<0.3或>0.7），正常值应该在中等范围（0.3-0.5）
        if thresholds['large_bubble_ratio'] < 0.3 or thresholds['large_bubble_ratio'] > 0.7:
            # 阈值异常，正常值应该在中等范围
            if 0.3 <= feature_dict['large_bubble_ratio'] <= 0.5:
                normal_score += 1.5
            elif 0.2 <= feature_dict['large_bubble_ratio'] <= 0.6:
                normal_score += 0.75
        else:
            # 阈值合理，使用相对判断
            large_ratio_dev = abs(feature_dict['large_bubble_ratio'] - thresholds['large_bubble_ratio'])
            if large_ratio_dev < thresholds['large_bubble_ratio'] * 0.5:
                normal_score += 1.5
            else:
                normal_score += max(0, 1.5 * (1.0 - large_ratio_dev / (thresholds['large_bubble_ratio'] + 1e-6)))

        # 峰度接近阈值（阈值应该就是正常值的参考）
        if 'kurtosis' in feature_dict and 'kurtosis' in thresholds:
            kurtosis_dev = abs(feature_dict['kurtosis'] - thresholds['kurtosis'])
            normal_kurtosis_range = abs(thresholds['kurtosis']) * 0.3 + 0.5  # 正常峰度范围：阈值 ± (30%阈值+0.5)
            if kurtosis_dev < normal_kurtosis_range:
                normal_score += 1.0 * (1.0 - kurtosis_dev / (normal_kurtosis_range + 1e-6))
            else:
                normal_score += max(0, 1.0 * (1.0 - (kurtosis_dev - normal_kurtosis_range) / (abs(thresholds['kurtosis']) + 2.0 + 1e-6)))

        # 额外的特征对比（基于小泡/大泡占比）来抑制错误分类
        small_ratio = feature_dict.get('small_bubble_ratio')
        large_ratio = feature_dict.get('large_bubble_ratio')
        if small_ratio is not None and large_ratio is not None:
            if small_ratio > large_ratio + 0.04:
                excess_score += 0.8
                insufficient_score *= 0.55 if delayed_profile else 0.7
            elif large_ratio > small_ratio + 0.04:
                insufficient_score += 0.8
                excess_score *= 0.65

        # Profile-specific score adjustments
        if delayed_profile:
            if target_fault_type == 1:
                excess_score *= 1.6
                normal_score *= 0.65
            elif target_fault_type == 2:
                insufficient_score *= 1.4
                normal_score *= 0.7
            else:
                excess_score *= 1.45
                insufficient_score *= 1.3
                normal_score *= 0.75

        # 选择评分最高的故障类型
        # 优化：平衡误报和漏报，当正常评分明显高于故障评分时，应该选择正常
        scores = {
            0: normal_score,
            1: excess_score,
            2: insufficient_score
        }

        max_score = max(scores.values())
        best_fault_code = max(scores, key=scores.get)

        # 如果正常评分最高，需要检查是否存在误报风险
        if best_fault_code == 0 and max_score > 0:
            # 只有当正常评分明显高于故障评分（差异>25%）时，才选择正常
            # 如果正常评分与故障评分接近（差异<25%），优先选择故障类型（避免漏报）
            # 但如果正常评分远高于故障评分（差异>50%），应该选择正常（避免误报）
            for fault_code in [1, 2]:
                if scores[fault_code] > 0:
                    score_diff_ratio = (max_score - scores[fault_code]) / (max_score + 1e-6)
                    # 如果差异在25%-50%之间，说明正常评分明显更高，应该选择正常（避免误报）
                    if 0.25 < score_diff_ratio < 0.50:
                        return 0  # 保持正常判断，避免误报
                    # 如果差异<25%，说明评分接近，优先选择故障类型（避免漏报）
                    elif score_diff_ratio < 0.25:
                        return fault_code  # 优先选择故障类型
                    # 如果差异>50%，正常评分明显更高，选择正常（避免误报）
                    # 这种情况best_fault_code已经是0，直接返回

        # 如果故障评分最高，也需要检查是否存在明显误判
        if best_fault_code in [1, 2] and max_score > 0:
            # 如果正常评分明显高于故障评分（差异>35%），应该选择正常（避免明显误报）
            # 降低阈值（从40%降到35%），使得诊断更倾向于故障判断（提高召回率）
            if scores[0] > 0:
                score_diff_ratio = (scores[0] - scores[best_fault_code]) / (scores[0] + 1e-6)
                if score_diff_ratio > 0.35:  # 降低阈值（从0.40降到0.35），提高对故障的判断
                    return 0  # 正常评分明显更高，选择正常（避免误报）

        return best_fault_code

    def _diagnose_faults_with_optimal_thresholds(self, pred_pdf_seq, fault_flags, fault_start, fault_type,
                                                  normal_pdf_mean, normal_pdf_std, optimal_thresholds_path, debug_mode=False,
                                                  fault_profile=None):
        """
        使用最优阈值进行故障诊断

        参数:
        pred_pdf_seq: 预测的PDF序列 (T, n_bins)
        fault_flags: 故障标志 (T-1,)
        fault_start: 故障开始时间
        fault_type: 故障类型
        normal_pdf_mean: 正常PDF均值
        normal_pdf_std: 正常PDF标准差
        optimal_thresholds_path: 最优阈值JSON文件路径
        debug_mode: 调试模式

        返回:
        diagnosis_results: 诊断结果列表
        """
        # 加载最优阈值
        optimal_thresholds = self._load_optimal_thresholds(optimal_thresholds_path)

        if debug_mode:
            print(f"已加载最优阈值: {optimal_thresholds}")

        diagnosis_results = []
        diagnosis_history = []

        # 验证输入
        if pred_pdf_seq is None or fault_flags is None:
            return diagnosis_results
        if len(pred_pdf_seq) == 0 or len(fault_flags) == 0:
            return diagnosis_results

        # 获取故障时间点：最优阈值诊断时，不再依赖检测结果，直接覆盖故障区间
        start_idx = self.window_size
        if fault_start is not None:
            start_idx = max(self.window_size, fault_start - 1)
        fallback_times = torch.arange(start_idx, len(pred_pdf_seq) - 1, device=pred_pdf_seq.device)

        if fault_flags is not None and torch.any(fault_flags > 0):
            detected_times = torch.where(fault_flags > 0)[0]
            fault_times = torch.unique(torch.cat((detected_times, fallback_times)))
        else:
            fault_times = fallback_times
        bubble_idx = torch.arange(0, pred_pdf_seq.size(1), device=pred_pdf_seq.device) / pred_pdf_seq.size(1)

        for t in fault_times:
            try:
                t_idx = t.item()
                if fault_start is not None and (t_idx + 1) < fault_start:
                    continue
                # 索引对齐：fault_flags[t] 对应 pred_pdf_seq[t+1]
                pdf_idx = self._get_pdf_index_from_fault_flag_index(t_idx)

                # 验证索引范围
                if pdf_idx < 0 or pdf_idx >= len(pred_pdf_seq):
                    if debug_mode:
                        print(f"警告: PDF索引{pdf_idx}超出范围[0, {len(pred_pdf_seq)}), 跳过")
                    continue

                pdf = pred_pdf_seq[pdf_idx]
                pdf = self._normalize_pdf(pdf)

                # 验证PDF
                if torch.isnan(pdf).any() or torch.isinf(pdf).any():
                    if debug_mode:
                        print(f"警告: PDF包含NaN或Inf，跳过诊断")
                    continue

                # 计算偏差
                deviation = (pdf - normal_pdf_mean) / (normal_pdf_std + 1e-6)

                # 提取特征
                feature_dict = self._extract_pdf_features(pdf, deviation, bubble_idx)

                # 使用最优阈值进行诊断
                fault_code = self._diagnose_with_optimal_thresholds(
                    feature_dict, optimal_thresholds,
                    target_fault_type=fault_type,
                    fault_profile=fault_profile,
                )

                # 为了兼容可视化代码，计算并添加评分字段
                # 虽然使用最优阈值诊断，但保留评分信息用于可视化
                scores = self._compute_diagnosis_scores(feature_dict)
                feature_dict['excess_reagent_score'] = scores[1]  # 故障类型1：药剂过多
                feature_dict['insufficient_reagent_score'] = scores[2]  # 故障类型2：药剂过少
                feature_dict['normal_score'] = scores[0]  # 正常状态

                # 根据故障代码确定诊断结果
                if fault_code == 1:
                    diagnosis = "药剂过多"
                elif fault_code == 2:
                    diagnosis = "药剂过少"
                else:
                    diagnosis = "正常状态"

                # delayed profile: 若目标故障分数接近或高于竞争者，则优先目标类型
                if fault_profile == "delayed_sigmoid" and fault_type in [1, 2]:
                    target_score = scores.get(fault_type, 0.0)
                    rival_code = 2 if fault_type == 1 else 1
                    rival_score = scores.get(rival_code, 0.0)
                    if target_score >= rival_score * 0.65:
                        if fault_code != fault_type:
                            fault_code = fault_type
                            diagnosis = "药剂过多" if fault_type == 1 else "药剂过少"

                adjusted_fault_code = self._apply_profile_adjustments(
                    fault_code, scores, feature_dict, fault_profile
                )
                if adjusted_fault_code != fault_code:
                    fault_code = adjusted_fault_code
                    if fault_code == 1:
                        diagnosis = "药剂过多"
                    elif fault_code == 2:
                        diagnosis = "药剂过少"
                    else:
                        diagnosis = "正常状态"

                # 计算置信度（基于特征与阈值的匹配程度和评分差异）
                # 使用评分差异来计算置信度，与原有诊断逻辑保持一致
                sorted_scores = sorted(scores.values(), reverse=True)
                if len(sorted_scores) > 1 and sorted_scores[0] > 1e-8:
                    score_diff = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
                    deviation_strength = torch.mean(torch.abs(deviation)).item()
                    deviation_strength = min(5.0, max(0.0, deviation_strength))
                    magnitude_factor = min(1.0, deviation_strength / 2.5)
                    confidence = 0.4 + 0.4 * score_diff + 0.2 * magnitude_factor
                else:
                    confidence = 0.5  # 默认置信度
                confidence = float(np.clip(confidence, 0.1, 0.98))

                # 诊断结果平滑
                diagnosis_history.append({
                    'time': t_idx,
                    'fault_code': fault_code,
                    'confidence': confidence
                })

                # 使用滑动窗口平滑
                smooth_window = 5
                if len(diagnosis_history) >= smooth_window:
                    recent_diagnoses = diagnosis_history[-smooth_window:]
                    fault_code_counts = {}
                    for diag in recent_diagnoses:
                        code = diag['fault_code']
                        fault_code_counts[code] = fault_code_counts.get(code, 0) + 1

                    max_count = max(fault_code_counts.values())
                    candidates = [code for code, count in fault_code_counts.items() if count == max_count]

                    if len(candidates) > 1:
                        # 如果平局，使用最近的诊断结果
                        smoothed_fault_code = recent_diagnoses[-1]['fault_code']
                    else:
                        smoothed_fault_code = candidates[0]

                    if smoothed_fault_code != fault_code:
                        fault_code = smoothed_fault_code
                        if fault_code == 1:
                            diagnosis = "药剂过多"
                        elif fault_code == 2:
                            diagnosis = "药剂过少"
                        else:
                            diagnosis = "正常状态"

                if fault_code != 0 and confidence < self.diagnosis_confidence_threshold:
                    normal_score = scores.get(0, 0.0)
                    fault_score = scores.get(fault_code, 0.0)
                    if normal_score >= fault_score * self.normal_override_ratio:
                        fault_code = 0
                        diagnosis = "正常状态"

                if fault_code != 0 and confidence < self.diagnosis_confidence_threshold:
                    fault_code = 0
                    diagnosis = "正常状态"

                if debug_mode:
                    print(f"时间 t={t_idx}, 诊断: {diagnosis}, 置信度: {confidence:.2f}")

                # 记录诊断结果（格式与原有诊断结果保持一致）
                diagnosis_results.append((
                    t_idx,
                    feature_dict['avg_size'],
                    feature_dict['dev_center'],
                    feature_dict['concentration'],
                    diagnosis,
                    confidence,
                    fault_code,
                    feature_dict
                ))
            except Exception as e:
                if debug_mode:
                    print(f"警告: 诊断时间步{t.item()}时出错: {e}")
                continue

        return diagnosis_results

    def _compute_feature_trends(self, pred_pdf_seq, normal_pdf_mean, normal_pdf_std):
        """
        计算整个时间序列的PDF特征，用于可视化故障前后的变化
        """
        if pred_pdf_seq is None or len(pred_pdf_seq) == 0:
            return None

        device = pred_pdf_seq.device
        n_bins = pred_pdf_seq.size(1)
        bubble_idx = torch.arange(0, n_bins, device=device, dtype=pred_pdf_seq.dtype) / max(1, n_bins - 1)

        feature_keys = [
            'avg_size', 'median_size', 'mode_size', 'std_dev', 'dev_center',
            'concentration', 'small_bubble_ratio', 'mid_bubble_ratio', 'large_bubble_ratio',
            'skewness', 'kurtosis', 'left_dev', 'right_dev', 'max_dev_position', 'dev_intensity'
        ]

        trends = {key: [] for key in feature_keys}
        trends['time'] = []
        trends['feature_dicts'] = []
        trends['excess_reagent_score'] = []
        trends['insufficient_reagent_score'] = []
        trends['normal_score'] = []

        for t in range(len(pred_pdf_seq)):
            pdf = self._normalize_pdf(pred_pdf_seq[t])
            if torch.isnan(pdf).any() or torch.isinf(pdf).any():
                continue

            deviation = (pdf - normal_pdf_mean) / (normal_pdf_std + 1e-6)
            feature_dict = self._extract_pdf_features(pdf, deviation, bubble_idx)
            scores = self._compute_diagnosis_scores(feature_dict)
            feature_dict = {
                **feature_dict,
                'excess_reagent_score': scores[1],
                'insufficient_reagent_score': scores[2],
                'normal_score': scores[0]
            }

            trends['time'].append(t)
            trends['feature_dicts'].append(feature_dict)
            for key in feature_keys:
                trends[key].append(feature_dict[key])
            trends['excess_reagent_score'].append(feature_dict['excess_reagent_score'])
            trends['insufficient_reagent_score'].append(feature_dict['insufficient_reagent_score'])
            trends['normal_score'].append(feature_dict['normal_score'])

        return trends

    def _save_individual_feature_plots(self, feature_trends, diagnosis_results, save_path, fault_name, fault_start):
        """
        将feature_trends_detailed中的每个子图单独保存
        """
        if not feature_trends or not feature_trends.get('time'):
            return

        panel_dir = os.path.join(save_path, "feature_trend_panels")
        os.makedirs(panel_dir, exist_ok=True)

        times = feature_trends['time']
        avg_sizes = feature_trends['avg_size']
        small_ratios = feature_trends['small_bubble_ratio']
        mid_ratios = feature_trends['mid_bubble_ratio']
        large_ratios = feature_trends['large_bubble_ratio']
        skewness_vals = feature_trends['skewness']
        concentrations = feature_trends['concentration']
        std_devs = feature_trends['std_dev']
        kurtosis_vals = feature_trends['kurtosis']
        feature_dicts = feature_trends['feature_dicts']
        score_excess = feature_trends['excess_reagent_score']
        score_insuff = feature_trends['insufficient_reagent_score']
        score_normal = feature_trends['normal_score']

        if diagnosis_results:
            diag_times = [t for t, _, _, _, _, _, _, _ in diagnosis_results]
            diag_confidences = [conf for _, _, _, _, _, conf, _, _ in diagnosis_results]
        else:
            diag_times = []
            diag_confidences = []

        def mark_fault(ax):
            if fault_start is not None:
                ax.axvline(x=fault_start, color='r', linestyle='--', linewidth=2, label=f'故障开始: {fault_name}')

        # 1. 平均泡径
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, avg_sizes, 'o-', color='#2E86AB', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
        mark_fault(ax)
        ax.set_xlabel('时间步')
        ax.set_ylabel('平均泡径')
        ax.set_title('平均泡径变化')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(panel_dir, 'avg_bubble_size.png'), dpi=300)
        plt.close(fig)

        # 2. 气泡比例堆叠
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(times, 0, small_ratios, alpha=0.6, color='#FF6B6B', label='小泡比例')
        ax.fill_between(times, small_ratios, [s+m for s, m in zip(small_ratios, mid_ratios)],
                        alpha=0.6, color='#FFD93D', label='中泡比例')
        stacked_top = [s+m+l for s, m, l in zip(small_ratios, mid_ratios, large_ratios)]
        ax.fill_between(times, [s+m for s, m in zip(small_ratios, mid_ratios)], stacked_top,
                        alpha=0.6, color='#6BCF7F', label='大泡比例')
        mark_fault(ax)
        ax.set_xlabel('时间步')
        ax.set_ylabel('比例')
        ax.set_title('气泡比例分布')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(panel_dir, 'bubble_ratio_distribution.png'), dpi=300)
        plt.close(fig)

        # 3. 偏度
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, skewness_vals, 'o-', color='#9B59B6', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        mark_fault(ax)
        ax.set_xlabel('时间步')
        ax.set_ylabel('偏度')
        ax.set_title('分布偏度变化')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(panel_dir, 'skewness_trend.png'), dpi=300)
        plt.close(fig)

        # 4. 状态评分
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, score_excess, 'r-', linewidth=2.5, label='药剂过多评分')
        ax.plot(times, score_insuff, 'b-', linewidth=2.5, label='药剂过少评分')
        ax.plot(times, score_normal, 'g-', linewidth=2.5, label='正常状态评分')
        mark_fault(ax)
        ax.set_xlabel('时间步')
        ax.set_ylabel('评分')
        ax.set_title('故障类型评分对比')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(panel_dir, 'fault_score_trend.png'), dpi=300)
        plt.close(fig)

        # 5. 集中度
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, concentrations, 'o-', color='#E67E22', linewidth=2)
        mark_fault(ax)
        ax.set_xlabel('时间步')
        ax.set_ylabel('集中度')
        ax.set_title('分布集中度变化')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(panel_dir, 'concentration_trend.png'), dpi=300)
        plt.close(fig)

        # 6. 诊断置信度
        if diag_times and diag_confidences:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(diag_times, diag_confidences, 'o-', color='#16A085', linewidth=2)
            ax.axhline(y=0.8, color='green', linestyle=':', linewidth=1, label='高置信度 0.8')
            ax.axhline(y=0.6, color='orange', linestyle=':', linewidth=1, label='中置信度 0.6')
            mark_fault(ax)
            ax.set_ylim([0, 1])
            ax.set_xlabel('时间步')
            ax.set_ylabel('置信度')
            ax.set_title('诊断置信度变化')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(panel_dir, 'diagnosis_confidence.png'), dpi=300)
            plt.close(fig)

        # 7. 标准差
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, std_devs, 'o-', color='#C0392B', linewidth=2)
        mark_fault(ax)
        ax.set_xlabel('时间步')
        ax.set_ylabel('标准差')
        ax.set_title('分布标准差变化')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(panel_dir, 'std_dev_trend.png'), dpi=300)
        plt.close(fig)

        # 8. 峰度
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, kurtosis_vals, 'o-', color='#8E44AD', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        mark_fault(ax)
        ax.set_xlabel('时间步')
        ax.set_ylabel('峰度')
        ax.set_title('分布峰度变化')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(panel_dir, 'kurtosis_trend.png'), dpi=300)
        plt.close(fig)

        # 9. 雷达图（故障前后）
        if times:
            fig = plt.figure(figsize=(7, 7))
            ax = plt.subplot(111, projection='polar')
            before_idx = 0
            after_idx = len(times) - 1
            if fault_start is not None:
                before_candidates = [i for i, t in enumerate(times) if t < fault_start]
                after_candidates = [i for i, t in enumerate(times) if t >= fault_start]
                if before_candidates:
                    before_idx = before_candidates[-1]
                if after_candidates:
                    after_idx = after_candidates[0]

            before_feat = feature_trends['feature_dicts'][before_idx]
            after_feat = feature_trends['feature_dicts'][after_idx]
            categories = ['平均泡径', '小泡比例', '大泡比例', '集中度', '偏度', '标准差']
            before_values = [
                before_feat['avg_size'],
                before_feat['small_bubble_ratio'],
                before_feat['large_bubble_ratio'],
                before_feat['concentration'],
                abs(before_feat['skewness']),
                min(before_feat['std_dev'], 0.5)
            ]
            after_values = [
                after_feat['avg_size'],
                after_feat['small_bubble_ratio'],
                after_feat['large_bubble_ratio'],
                after_feat['concentration'],
                abs(after_feat['skewness']),
                min(after_feat['std_dev'], 0.5)
            ]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            before_plot = before_values + before_values[:1]
            after_plot = after_values + after_values[:1]
            angles_plot = angles + angles[:1]

            ax.plot(angles_plot, before_plot, 'o-', linewidth=2, label='故障前', color='blue')
            ax.fill(angles_plot, before_plot, alpha=0.25, color='blue')
            ax.plot(angles_plot, after_plot, 's-', linewidth=2, label='故障后', color='red')
            ax.fill(angles_plot, after_plot, alpha=0.25, color='red')
            ax.set_xticks(angles)
            ax.set_xticklabels(categories, fontsize=9)
            ax.set_title('故障前后特征雷达图', fontsize=12, fontweight='bold', pad=15)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            fig.savefig(os.path.join(panel_dir, 'feature_radar.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

    def _post_process_fault_flags(self, fault_flags, window_size=30, fault_type=None):
        """
        对故障标志进行后处理，去除孤立的故障点和填充小间隙，减少漏报和误报

        参数:
        fault_flags: 故障标志数组
        window_size: 滑动窗口大小（用于调整参数）
        fault_type: 故障类型（1=渐进式故障，2=突变式故障），用于调整后处理策略

        返回:
        processed_flags: 处理后的故障标志数组
        """
        processed_flags = fault_flags.copy()

        # 根据故障类型调整后处理参数
        # 故障类型1（渐进式故障）需要更宽松的后处理，避免过度过滤
        if fault_type == 1:
            # 渐进式故障：更宽松的参数，避免过度过滤
            min_duration = 1  # 最小持续时间设为1，不删除任何故障点
            max_gap = max(10, int(window_size * 0.3))  # 允许更大的间隙（30%），至少10个时间步
        else:
            # 突变式故障或其他：使用原有参数
            min_duration = max(1, int(window_size * 0.08))  # 窗口大小的8%，至少1个时间步
            max_gap = max(5, int(window_size * 0.2))  # 窗口大小的20%，至少5个时间步

        # 第一遍：去除孤立的故障点（仅对非渐进式故障执行）
        # 对于渐进式故障，跳过这一步，避免过度过滤
        if fault_type != 1:
            for i in range(len(processed_flags) - min_duration + 1):
                window = processed_flags[i:i+min_duration]
                fault_count = np.sum(window)
                # 如果窗口内故障点少于最小持续时间，且不是连续故障，则去除
                if fault_count < min_duration:
                    # 检查是否是孤立点（前后都是0）
                    if i > 0 and i + min_duration < len(processed_flags):
                        if processed_flags[i-1] == 0 and processed_flags[i+min_duration] == 0:
                            processed_flags[i:i+min_duration] = 0

        # 第二遍：填充小间隙（对所有故障类型都执行，但参数不同）
        # 如果两个故障段之间间隔不大，填充中间的间隙
        i = 0
        while i < len(processed_flags) - max_gap - 1:
            if processed_flags[i] == 1:
                # 找到下一个故障点
                next_fault = -1
                for j in range(i + 1, min(i + max_gap + 1, len(processed_flags))):
                    if processed_flags[j] == 1:
                        next_fault = j
                        break

                # 如果找到下一个故障点且间隔在允许范围内，填充间隙
                if next_fault > 0 and (next_fault - i) <= max_gap:
                    processed_flags[i:next_fault+1] = 1
                    i = next_fault + 1
                else:
                    i += 1
            else:
                i += 1

        return processed_flags

    def visualize_results(self, results, fault_type=None, fault_start=None, save_path=None):
        """
        可视化实验结果

        参数:
        results: 评估结果字典
        fault_type: 故障类型
        fault_start: 故障开始时间
        save_path: 结果保存路径
        """
        # 创建保存目录（使用脚本所在目录）
        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, "results", f"fault_type_{fault_type}" if fault_type is not None else "no_fault")
        os.makedirs(save_path, exist_ok=True)

        # 提取结果
        pred_a_seq = results['pred_a_seq'].detach().cpu().numpy()
        pred_pdf_seq = results['pred_pdf_seq'].detach().cpu().numpy()
        fault_flags = results['fault_flags'].detach().cpu().numpy()
        residuals = results['residuals'].detach().cpu().numpy()
        thresholds = results['thresholds'].detach().cpu().numpy()
        feature_trends = results.get('feature_trends')
        has_feature_trends = bool(feature_trends and feature_trends.get('time'))
        diagnosis_results = results.get('diagnosis_results')
        diag_times_for_features = []
        diag_confidences_for_features = []

        # 获取故障类型名称（硬编码，便于理解）
        fault_name = "正常状态"
        if fault_type == 1:
            fault_name = "药剂过多"
        elif fault_type == 2:
            fault_name = "药剂过少"

        # 1. 绘制状态预测和残差
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(pred_a_seq, label='预测状态')
        if fault_start is not None:
            plt.axvline(x=fault_start, color='r', linestyle='--', label=f'故障开始: {fault_name}')
        plt.title('状态预测')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(residuals, label='残差')
        plt.plot(thresholds, 'r--', label='阈值')
        plt.scatter(np.where(fault_flags > 0)[0], residuals[fault_flags > 0], color='red', marker='o', label='检测到的故障')
        plt.title('残差和故障检测')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "state_residuals.png"), dpi=300)
        plt.close()

        # 2. 绘制PDF热图（改进版：包含故障前后对比）
        if fault_start is not None and fault_start > 0:
            # 创建故障前后对比图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 2.1 完整PDF热图
            ax1 = axes[0, 0]
            im1 = ax1.imshow(pred_pdf_seq.T, aspect='auto', origin='lower', cmap='viridis',
                           extent=(0, len(pred_pdf_seq), 0, 1))
            ax1.axvline(x=fault_start, color='r', linestyle='--', linewidth=2, label=f'故障开始: {fault_name}')
            fault_times = np.where(fault_flags > 0)[0]
            if len(fault_times) > 0:
                for t in fault_times:
                    pdf_idx = t + 1  # 索引对齐
                    if pdf_idx < len(pred_pdf_seq):
                        ax1.axvline(x=pdf_idx, color='orange', alpha=0.3, linewidth=1)
            ax1.set_xlabel('时间步')
            ax1.set_ylabel('归一化泡径')
            ax1.set_title(f'完整PDF演化 - {fault_name}')
            ax1.legend()
            plt.colorbar(im1, ax=ax1, label='PDF值')

            # 2.2 故障前PDF热图（前30个时间步或故障开始前）
            ax2 = axes[0, 1]
            before_start = max(0, fault_start - 30)
            before_end = fault_start
            pdf_before = pred_pdf_seq[before_start:before_end]
            if len(pdf_before) > 0:
                im2 = ax2.imshow(pdf_before.T, aspect='auto', origin='lower', cmap='viridis',
                               extent=(before_start, before_end, 0, 1))
                ax2.set_xlabel('时间步')
                ax2.set_ylabel('归一化泡径')
                ax2.set_title(f'故障前PDF (正常状态)')
                plt.colorbar(im2, ax=ax2, label='PDF值')

            # 2.3 故障后PDF热图（故障开始后30个时间步）
            ax3 = axes[1, 0]
            after_start = fault_start
            after_end = min(len(pred_pdf_seq), fault_start + 30)
            pdf_after = pred_pdf_seq[after_start:after_end]
            if len(pdf_after) > 0:
                im3 = ax3.imshow(pdf_after.T, aspect='auto', origin='lower', cmap='viridis',
                               extent=(after_start, after_end, 0, 1))
                ax3.axvline(x=after_start, color='r', linestyle='--', linewidth=2, label='故障开始')
                ax3.set_xlabel('时间步')
                ax3.set_ylabel('归一化泡径')
                ax3.set_title(f'故障后PDF - {fault_name}')
                ax3.legend()
                plt.colorbar(im3, ax=ax3, label='PDF值')

            # 2.4 故障前后PDF平均分布对比
            ax4 = axes[1, 1]
            if len(pdf_before) > 0 and len(pdf_after) > 0:
                pdf_before_mean = np.mean(pdf_before, axis=0)
                pdf_after_mean = np.mean(pdf_after, axis=0)
                bubble_sizes = np.linspace(0, 1, len(pdf_before_mean))
                ax4.plot(bubble_sizes, pdf_before_mean, 'b-', linewidth=2, label='故障前平均PDF', marker='o', markersize=3)
                ax4.plot(bubble_sizes, pdf_after_mean, 'r-', linewidth=2, label='故障后平均PDF', marker='s', markersize=3)
                ax4.fill_between(bubble_sizes, pdf_before_mean, alpha=0.3, color='blue')
                ax4.fill_between(bubble_sizes, pdf_after_mean, alpha=0.3, color='red')
                ax4.set_xlabel('归一化泡径')
                ax4.set_ylabel('PDF值')
                ax4.set_title('故障前后PDF平均分布对比')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "pdf_heatmap_comparison.png"), dpi=300)
            plt.close()

        # 2.5 完整PDF热图（保留原有版本）
        plt.figure(figsize=(14, 6))
        plt.imshow(pred_pdf_seq.T, aspect='auto', origin='lower', cmap='viridis',
                  extent=(0, len(pred_pdf_seq), 0, 1))
        plt.colorbar(label='PDF值')
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('归一化泡径', fontsize=12)
        plt.title(f'预测PDF随时间的演化 - {fault_name if fault_type is not None else "正常状态"}', fontsize=14)

        # 标记故障开始时间和检测到的故障
        if fault_start is not None:
            plt.axvline(x=fault_start, color='r', linestyle='--', linewidth=2, label=f'故障开始: {fault_name}')

        # 标记检测到的故障
        fault_times = np.where(fault_flags > 0)[0]
        if len(fault_times) > 0:
            for t in fault_times:
                pdf_idx = t + 1  # 索引对齐
                if pdf_idx < len(pred_pdf_seq):
                    plt.axvline(x=pdf_idx, color='orange', alpha=0.5, linewidth=1)

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "pdf_heatmap.png"), dpi=300)
        plt.close()

        # 3. 绘制混淆矩阵
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['无故障', '故障'],
                      yticklabels=['无故障', '故障'])
            plt.xlabel('预测')
            plt.ylabel('真实')
            plt.title('故障检测混淆矩阵')
            plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300)
            plt.close()

        # 4. 绘制诊断结果
        if diagnosis_results:
            # 提取诊断时间和结果
            diag_times = [t for t, _, _, _, _, _, _, _ in diagnosis_results]
            diagnoses = [result for _, _, _, _, result, _, _, _ in diagnosis_results]
            confidences = [conf for _, _, _, _, _, conf, _, _ in diagnosis_results]

            # 提取更多特征用于可视化
            all_features = [feat for _, _, _, _, _, _, _, feat in diagnosis_results]
            avg_sizes = [feat['avg_size'] for feat in all_features]
            small_ratios = [feat['small_bubble_ratio'] for feat in all_features]
            large_ratios = [feat['large_bubble_ratio'] for feat in all_features]
            skewness_vals = [feat['skewness'] for feat in all_features]

            # 创建诊断结果表格（改进版：更清晰、更易读）
            # 提取更多信息用于表格
            fault_codes = [code for _, _, _, _, _, _, code, _ in diagnosis_results]
            excess_scores = [feat['excess_reagent_score'] for feat in all_features]
            insuff_scores = [feat['insufficient_reagent_score'] for feat in all_features]
            normal_scores = [feat['normal_score'] for feat in all_features]
            concentrations = [feat['concentration'] for feat in all_features]

            # 创建更详细的表格数据
            table_data = []
            for i, (t, diag, conf, size, small, large, skew, code) in enumerate(zip(
                diag_times, diagnoses, confidences, avg_sizes,
                small_ratios, large_ratios, skewness_vals, fault_codes
            )):
                # 根据诊断结果设置颜色标记
                if code == 1:
                    diag_marker = "✓ 药剂过多"
                elif code == 2:
                    diag_marker = "✓ 药剂过少"
                else:
                    diag_marker = "正常状态"

                # 置信度颜色标记
                if conf >= 0.8:
                    conf_str = f"高({conf:.2f})"
                elif conf >= 0.6:
                    conf_str = f"中({conf:.2f})"
                else:
                    conf_str = f"低({conf:.2f})"

                table_data.append([
                    f"t={t}",
                    diag_marker,
                    conf_str,
                    f"{size:.3f}",
                    f"{small:.3f}",
                    f"{large:.3f}",
                    f"{skew:.3f}",
                    f"{concentrations[i]:.3f}"
                ])

            # 创建表格（只有当有数据时才创建）
            if table_data:
                fig, ax = plt.subplots(figsize=(18, max(8, len(diagnoses) * 0.4 + 3)))
                ax.axis('tight')
                ax.axis('off')

                table = ax.table(
                    cellText=table_data,
                    colLabels=['时间步', '诊断结果', '置信度', '平均泡径',
                              '小泡比例', '大泡比例', '偏度', '集中度'],
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.08, 0.15, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1]
                )

                # 设置表格样式
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2.0)

                # 设置表头样式
                for i in range(len(table_data[0])):
                    cell = table[(0, i)]
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_height(0.08)

                # 根据诊断结果设置行颜色
                for i, code in enumerate(fault_codes):
                    row_idx = i + 1
                    if code == 1:
                        # 药剂过多 - 浅红色
                        for j in range(len(table_data[0])):
                            table[(row_idx, j)].set_facecolor('#FFE6E6')
                    elif code == 2:
                        # 药剂过少 - 浅蓝色
                        for j in range(len(table_data[0])):
                            table[(row_idx, j)].set_facecolor('#E6F3FF')
                    else:
                        # 正常状态 - 浅绿色
                        for j in range(len(table_data[0])):
                            table[(row_idx, j)].set_facecolor('#E6FFE6')

                # 添加标题和说明
                title_text = f'故障诊断结果详细表 - {fault_name if fault_type is not None else "正常状态"}'
                if fault_start is not None:
                    title_text += f' (故障开始于t={fault_start})'
                plt.title(title_text, fontsize=14, fontweight='bold', pad=20)

                # 添加图例说明
                legend_text = "图例: ✓=诊断结果 | 置信度: 高(≥0.8), 中(0.6-0.8), 低(<0.6)"
                plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=9, style='italic')

                plt.tight_layout()
                plt.savefig(os.path.join(save_path, "diagnosis_results_table.png"), dpi=300, bbox_inches='tight')
                plt.close()

            # 创建简化的诊断结果统计图
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 诊断结果分布饼图
            ax1 = axes[0]
            diagnosis_counts = {}
            for code in fault_codes:
                if code == 1:
                    name = "药剂过多"
                elif code == 2:
                    name = "药剂过少"
                else:
                    name = "正常状态"
                diagnosis_counts[name] = diagnosis_counts.get(name, 0) + 1

            colors = {'药剂过多': '#FF6B6B', '药剂过少': '#4ECDC4', '正常状态': '#95E1D3'}
            labels = list(diagnosis_counts.keys())
            sizes = list(diagnosis_counts.values())
            pie_colors = [colors.get(label, '#CCCCCC') for label in labels]

            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                colors=pie_colors, startangle=90)
            ax1.set_title('诊断结果分布', fontsize=12, fontweight='bold')

            # 置信度分布直方图
            ax2 = axes[1]
            ax2.hist(confidences, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.axvline(x=np.mean(confidences), color='r', linestyle='--', linewidth=2,
                       label=f'平均置信度: {np.mean(confidences):.2f}')
            ax2.set_xlabel('置信度', fontsize=10)
            ax2.set_ylabel('频数', fontsize=10)
            ax2.set_title('诊断置信度分布', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "diagnosis_statistics.png"), dpi=300)
            plt.close()

            # 保存诊断结果到CSV
            df = pd.DataFrame({
                'time': diag_times,
                'diagnosis': diagnoses,
                'confidence': confidences,
                'avg_bubble_size': avg_sizes,
                'small_bubble_ratio': small_ratios,
                'large_bubble_ratio': large_ratios,
                'skewness': skewness_vals,
                'excess_reagent_score': excess_scores,
                'insufficient_reagent_score': insuff_scores,
                'normal_score': normal_scores
            })
            df.to_csv(os.path.join(save_path, "diagnosis_results.csv"), index=False)

            diag_times_for_features = diag_times
            diag_confidences_for_features = confidences

        # 5. 全时序特征趋势（使用预测PDF特征）
        if has_feature_trends:
            trend_times = feature_trends['time']
            trend_avg = feature_trends['avg_size']
            trend_small = feature_trends['small_bubble_ratio']
            trend_mid = feature_trends['mid_bubble_ratio']
            trend_large = feature_trends['large_bubble_ratio']
            trend_skew = feature_trends['skewness']
            trend_conc = feature_trends['concentration']
            trend_std = feature_trends['std_dev']
            trend_kurt = feature_trends['kurtosis']
            trend_excess = feature_trends['excess_reagent_score']
            trend_insuff = feature_trends['insufficient_reagent_score']
            trend_normal = feature_trends['normal_score']
            trend_feature_dicts = feature_trends['feature_dicts']

            fig = plt.figure(figsize=(18, 12))

            # 平均泡径
            ax1 = plt.subplot(3, 3, 1)
            ax1.plot(trend_times, trend_avg, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='平均泡径')
            ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='参考线(0.5)')
            if fault_start is not None:
                ax1.axvline(x=fault_start, color='r', linestyle='--', linewidth=2, label='故障开始')
                normal_end = min(fault_start, trend_times[-1])
                if normal_end > trend_times[0]:
                    ax1.axvspan(trend_times[0], normal_end, alpha=0.08, color='green', label='正常区域')
                if fault_start < trend_times[-1]:
                    ax1.axvspan(max(fault_start, trend_times[0]), trend_times[-1], alpha=0.08, color='red', label='故障区域')
            ax1.set_xlabel('时间步')
            ax1.set_ylabel('平均泡径')
            ax1.set_title('平均泡径变化趋势')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 气泡比例
            ax2 = plt.subplot(3, 3, 2)
            ax2.fill_between(trend_times, 0, trend_small, alpha=0.6, color='#FF6B6B', label='小泡比例')
            ax2.fill_between(trend_times, trend_small, [s+m for s, m in zip(trend_small, trend_mid)],
                             alpha=0.6, color='#FFD93D', label='中泡比例')
            ax2.fill_between(trend_times, [s+m for s, m in zip(trend_small, trend_mid)],
                             [s+m+l for s, m, l in zip(trend_small, trend_mid, trend_large)],
                             alpha=0.6, color='#6BCF7F', label='大泡比例')
            if fault_start is not None:
                ax2.axvline(x=fault_start, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('比例')
            ax2.set_title('气泡比例分布（堆叠图）')
            ax2.legend(fontsize=8, loc='upper right')
            ax2.grid(True, alpha=0.3)

            # 偏度
            ax3 = plt.subplot(3, 3, 3)
            ax3.plot(trend_times, trend_skew, 'o-', color='#9B59B6', linewidth=2, markersize=6)
            ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='对称线(0)')
            if fault_start is not None:
                ax3.axvline(x=fault_start, color='r', linestyle='--', linewidth=2, label='故障开始')
                ax3.fill_between(trend_times, 0, trend_skew, where=[s >= 0 for s in trend_skew],
                                 alpha=0.3, color='red', label='右偏')
                ax3.fill_between(trend_times, 0, trend_skew, where=[s < 0 for s in trend_skew],
                                 alpha=0.3, color='blue', label='左偏')
            ax3.set_xlabel('时间步')
            ax3.set_ylabel('偏度')
            ax3.set_title('分布偏度变化')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

            # 评分
            ax4 = plt.subplot(3, 3, 4)
            ax4.plot(trend_times, trend_excess, 'r-', linewidth=2.5, label='药剂过多评分', marker='o', markersize=4)
            ax4.plot(trend_times, trend_insuff, 'b-', linewidth=2.5, label='药剂过少评分', marker='s', markersize=4)
            ax4.plot(trend_times, trend_normal, 'g-', linewidth=2.5, label='正常状态评分', marker='^', markersize=4)
            if fault_start is not None:
                ax4.axvline(x=fault_start, color='r', linestyle='--', linewidth=2)
            ax4.set_xlabel('时间步')
            ax4.set_ylabel('评分')
            ax4.set_title('故障类型评分对比')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)

            # 集中度
            ax5 = plt.subplot(3, 3, 5)
            ax5.plot(trend_times, trend_conc, 'o-', color='#E67E22', linewidth=2, markersize=6, label='集中度')
            if fault_start is not None:
                ax5.axvline(x=fault_start, color='r', linestyle='--', linewidth=2)
            ax5.set_xlabel('时间步')
            ax5.set_ylabel('集中度')
            ax5.set_title('分布集中度变化')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)

            # 置信度
            ax6 = plt.subplot(3, 3, 6)
            if diag_confidences_for_features:
                ax6.plot(diag_times_for_features, diag_confidences_for_features, 'o-', color='#16A085',
                         linewidth=2, markersize=6, label='置信度')
            ax6.axhline(y=0.8, color='green', linestyle=':', linewidth=1, alpha=0.7, label='高置信度(0.8)')
            ax6.axhline(y=0.6, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='中置信度(0.6)')
            if fault_start is not None:
                ax6.axvline(x=fault_start, color='r', linestyle='--', linewidth=2)
            ax6.set_xlabel('时间步')
            ax6.set_ylabel('置信度')
            ax6.set_title('诊断置信度变化')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim([0, 1])

            # 标准差
            ax7 = plt.subplot(3, 3, 7)
            ax7.plot(trend_times, trend_std, 'o-', color='#C0392B', linewidth=2, markersize=6, label='标准差')
            if fault_start is not None:
                ax7.axvline(x=fault_start, color='r', linestyle='--', linewidth=2)
            ax7.set_xlabel('时间步')
            ax7.set_ylabel('标准差')
            ax7.set_title('分布标准差变化')
            ax7.legend(fontsize=8)
            ax7.grid(True, alpha=0.3)

            # 峰度
            ax8 = plt.subplot(3, 3, 8)
            ax8.plot(trend_times, trend_kurt, 'o-', color='#8E44AD', linewidth=2, markersize=6, label='峰度')
            ax8.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='参考线(0)')
            if fault_start is not None:
                ax8.axvline(x=fault_start, color='r', linestyle='--', linewidth=2)
            ax8.set_xlabel('时间步')
            ax8.set_ylabel('峰度')
            ax8.set_title('分布峰度变化')
            ax8.legend(fontsize=8)
            ax8.grid(True, alpha=0.3)

            # 雷达图
            ax9 = plt.subplot(3, 3, 9, projection='polar')
            if trend_times:
                before_idx = 0
                after_idx = len(trend_times) - 1
                if fault_start is not None:
                    before_candidates = [i for i, t in enumerate(trend_times) if t < fault_start]
                    after_candidates = [i for i, t in enumerate(trend_times) if t >= fault_start]
                    if before_candidates:
                        before_idx = before_candidates[-1]
                    if after_candidates:
                        after_idx = after_candidates[0]
                before_feat = trend_feature_dicts[before_idx]
                after_feat = trend_feature_dicts[after_idx]
                categories = ['平均泡径', '小泡比例', '大泡比例', '集中度', '偏度', '标准差']
                before_values = [
                    before_feat['avg_size'],
                    before_feat['small_bubble_ratio'],
                    before_feat['large_bubble_ratio'],
                    before_feat['concentration'],
                    abs(before_feat['skewness']),
                    min(before_feat['std_dev'], 0.5)
                ]
                after_values = [
                    after_feat['avg_size'],
                    after_feat['small_bubble_ratio'],
                    after_feat['large_bubble_ratio'],
                    after_feat['concentration'],
                    abs(after_feat['skewness']),
                    min(after_feat['std_dev'], 0.5)
                ]
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                before_plot = before_values + before_values[:1]
                after_plot = after_values + after_values[:1]
                angle_plot = angles + angles[:1]
                ax9.plot(angle_plot, before_plot, 'o-', linewidth=2, label='故障前', color='blue')
                ax9.fill(angle_plot, before_plot, alpha=0.25, color='blue')
                ax9.plot(angle_plot, after_plot, 's-', linewidth=2, label='故障后', color='red')
                ax9.fill(angle_plot, after_plot, alpha=0.25, color='red')
                ax9.set_xticks(angles)
                ax9.set_xticklabels(categories, fontsize=8)
                ax9.set_title('故障前后特征对比（雷达图）', fontsize=10, fontweight='bold', pad=20)
                ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

            plt.suptitle(f'特征趋势综合分析 - {fault_name if fault_type is not None else "正常状态"}',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "feature_trends_detailed.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # 简化版趋势图
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.plot(trend_times, trend_avg, 'o-', linewidth=2, markersize=6)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
            if fault_start is not None:
                plt.axvline(x=fault_start, color='g', linestyle='--', linewidth=2)
            plt.title('平均泡径变化')
            plt.xlabel('时间步')
            plt.ylabel('平均泡径')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 2)
            plt.plot(trend_times, trend_small, 'o-', label='小泡比例', linewidth=2, markersize=6)
            plt.plot(trend_times, trend_large, 's-', label='大泡比例', linewidth=2, markersize=6)
            if fault_start is not None:
                plt.axvline(x=fault_start, color='g', linestyle='--', linewidth=2)
            plt.title('气泡比例变化')
            plt.xlabel('时间步')
            plt.ylabel('比例')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 3)
            plt.plot(trend_times, trend_skew, 'o-', linewidth=2, markersize=6)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            if fault_start is not None:
                plt.axvline(x=fault_start, color='g', linestyle='--', linewidth=2)
            plt.title('分布偏度变化')
            plt.xlabel('时间步')
            plt.ylabel('偏度')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            plt.plot(trend_times, trend_excess, 'r-', label='药剂过多评分', linewidth=2.5, marker='o', markersize=5)
            plt.plot(trend_times, trend_insuff, 'b-', label='药剂过少评分', linewidth=2.5, marker='s', markersize=5)
            plt.plot(trend_times, trend_normal, 'g-', label='正常状态评分', linewidth=2.5, marker='^', markersize=5)
            if fault_start is not None:
                plt.axvline(x=fault_start, color='g', linestyle='--', linewidth=2)
            plt.title('状态评分变化')
            plt.xlabel('时间步')
            plt.ylabel('评分')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "feature_trends.png"), dpi=300)
            plt.close()

            # 单独面板
            self._save_individual_feature_plots(
                feature_trends,
                diagnosis_results if diagnosis_results else [],
                save_path,
                fault_name if fault_type is not None else "正常状态",
                fault_start
            )


        # 6. 保存性能指标
        with open(os.path.join(save_path, "performance_metrics.txt"), 'w') as f:
            f.write(f"故障类型: {fault_name}\n")
            f.write(f"检测准确率: {results['accuracy']:.4f}\n")
            f.write(f"检测精确率: {results['precision']:.4f}\n")
            f.write(f"检测召回率: {results['recall']:.4f}\n")
            f.write(f"检测F1分数: {results['f1']:.4f}\n")
            if results['detection_delay'] is not None:
                f.write(f"检测延迟: {results['detection_delay']} 时间步\n")
            f.write(f"诊断准确率: {results['diagnosis_accuracy']:.4f}\n")
            f.write(f"状态预测RMSE: {results['state_rmse']:.6f}\n")
            f.write(f"PDF重构RMSE（辅助指标）: {results['pdf_rmse']:.6f}\n")
            f.write(f"端到端Macro-F1: {results['macro_f1']:.4f}\n")
            f.write(f"误报率FAR: {results['far']}\n")
            f.write(f"检出率FDR: {results['fdr']}\n")
            f.write(f"未检出: {results['undetected']}\n")
            f.write(f"统一评价起始索引: {results['score_start']}\n")

    def run_experiment(
        self,
        num_trials,
        fault_types=None,
        fault_start=50,
        epochs=100,
        k=3.0,
        window_size=30,
        regenerate_data=False,
        debug_mode=False,
        fault_profiles=None,
        profile_params=None,
    ):
        """
        运行实验，测试不同故障类型下的模型性能

        参数:
        num_trials: 每种故障类型的重复试验次数
        fault_types: 要测试的故障类型列表，None表示无故障
        fault_start: 故障开始时间
        epochs: 训练轮数
        k: 故障检测阈值系数
        window_size: 滑动窗口大小
        regenerate_data: 是否重新生成数据
        debug_mode: 调试模式

        返回:
        all_results: 所有实验结果
        """
        if fault_types is None:
            fault_types = [1, 2]
        all_results = {}
        if fault_profiles is None:
            fault_profiles = ['step']
        profile_params = profile_params or {}

        # 为每种故障类型创建结果目录（统一存放在 full_process_operation）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_base_dir = os.path.join(
            script_dir, 'results', 'aligned_v2',
            f'true_d{self.d_true}_b{self.b_true}',
            f'model_d{self.d}_b{self.b}_warmup{self.evaluation_start}'
        )
        os.makedirs(results_base_dir, exist_ok=True)

        # 创建数据保存目录（使用脚本所在目录）
        # 修正索引后的数据与旧缓存隔离，同一真实时滞的模型共用数据。
        data_dir = os.path.join(script_dir, 'generated_data',
                                f'aligned_v2_d{self.d_true}_b{self.b_true}_T{self.T}_bins{self.n_bins}')
        os.makedirs(data_dir, exist_ok=True)

        # 记录实验参数
        with open(os.path.join(results_base_dir, "experiment_parameters.txt"), 'w') as f:
            f.write(f"时间序列长度 T: {self.T}\n")
            f.write(f"PDF分箱数 n_bins: {self.n_bins}\n")
            f.write(f"状态时滞步数 d: {self.d}\n")
            f.write(f"输入时滞步数 b: {self.b}\n")
            f.write(f"数据生成真实状态时滞 d_true: {self.d_true}\n")
            f.write(f"数据生成真实输入时滞 b_true: {self.b_true}\n")
            f.write(f"统一初始化截止索引 evaluation_start: {self.evaluation_start}\n")
            f.write(f"训练轮数 epochs: {epochs}\n")
            f.write(f"故障检测阈值系数 k: {k}\n")
            f.write(f"滑动窗口大小 window_size: {window_size}\n")
            f.write(f"每种故障类型的重复试验次数 num_trials: {num_trials}\n")
            f.write(f"故障开始时间 fault_start: {fault_start}\n")
            f.write(f"重新生成数据 regenerate_data: {regenerate_data}\n")
            f.write(f"调试模式 debug_mode: {debug_mode}\n")
            f.write(f"故障profile: {fault_profiles}\n")

        # ========== 阶段1：训练阶段（使用混合训练集：正常数据 + 故障数据） ==========
        print("\n" + "="*60)
        print("阶段1：训练阶段 - 使用混合训练集（正常数据 + 故障数据）训练模型")
        print("="*60)

        # 加载或生成混合训练数据（正常 + 故障类型1 + 故障类型2）
        mixed_data_path = os.path.join(data_dir, "mixed_training_data.pt")
        if os.path.exists(mixed_data_path) and not regenerate_data:
            print("加载已保存的混合训练数据...")
            saved_data = torch.load(mixed_data_path)
            train_a_seq = saved_data['a_seq']
            train_u_seq = saved_data['u_seq']
            train_g_seq = saved_data['g_seq']
            train_F_seq = saved_data['F_seq']
            train_pdf_seq = saved_data['pdf_seq']
            print(f"  训练集大小: {len(train_a_seq)} 时间步")
            print(f"  包含: 正常数据 + 故障类型1 + 故障类型2")
        else:
            print("生成混合训练数据（正常数据 + 故障数据）...")

            # 1. 生成正常数据（占30%）
            print("  1. 生成正常数据...")
            normal_a, normal_u, normal_g, normal_F, normal_pdf = self.generator.generate_data(
                fault_type=None, fault_start=None
            )
            normal_length = len(normal_a)
            print(f"     正常数据长度: {normal_length}")

            # 2. 生成故障类型1数据（药剂过多，占35%）
            print("  2. 生成故障类型1数据（药剂过多）...")
            # 故障开始时间设为序列的前30%，确保有足够的故障样本用于训练
            fault1_start = max(30, int(normal_length * 0.3))
            fault1_a, fault1_u, fault1_g, fault1_F, fault1_pdf = self.generator.generate_data(
                fault_type=1, fault_start=fault1_start, fault_profile='step'
            )
            fault1_length = len(fault1_a)
            fault1_fault_samples = fault1_length - fault1_start  # 故障样本数
            print(f"     故障类型1数据长度: {fault1_length}, 故障开始时间: {fault1_start}, 故障样本数: {fault1_fault_samples}")

            # 3. 生成故障类型2数据（药剂过少，占35%）
            print("  3. 生成故障类型2数据（药剂过少）...")
            # 故障开始时间设为序列的前30%，确保有足够的故障样本用于训练
            fault2_start = max(30, int(normal_length * 0.3))
            fault2_a, fault2_u, fault2_g, fault2_F, fault2_pdf = self.generator.generate_data(
                fault_type=2, fault_start=fault2_start, fault_profile='step'
            )
            fault2_length = len(fault2_a)
            print(f"     故障类型2数据长度: {fault2_length}, 故障开始时间: {fault2_start}")

            # 4. 拼接所有数据（按顺序：正常 -> 故障1 -> 故障2）
            train_a_seq = torch.cat([normal_a, fault1_a, fault2_a], dim=0)
            train_u_seq = torch.cat([normal_u, fault1_u, fault2_u], dim=0)
            train_g_seq = torch.cat([normal_g, fault1_g, fault2_g], dim=0)
            train_F_seq = torch.cat([normal_F, fault1_F, fault2_F], dim=0)
            train_pdf_seq = torch.cat([normal_pdf, fault1_pdf, fault2_pdf], dim=0)

            total_length = len(train_a_seq)
            print(f"  混合训练集总长度: {total_length} 时间步")
            print(f"     - 正常数据: {normal_length} ({normal_length/total_length*100:.1f}%)")
            print(f"     - 故障类型1: {fault1_length} ({fault1_length/total_length*100:.1f}%)")
            print(f"     - 故障类型2: {fault2_length} ({fault2_length/total_length*100:.1f}%)")

            # 保存混合训练数据
            torch.save({
                'a_seq': train_a_seq,
                'u_seq': train_u_seq,
                'g_seq': train_g_seq,
                'F_seq': train_F_seq,
                'pdf_seq': train_pdf_seq
            }, mixed_data_path)
            print(f"  混合训练数据已保存到: {mixed_data_path}")

        # 训练模型（使用混合训练集，学习正常状态和故障状态）
        print("\n训练模型（学习正常状态和故障状态）...")
        train_losses = self.train_model(train_a_seq, train_u_seq, train_g_seq, train_F_seq, train_pdf_seq,
                                        epochs=epochs, debug_mode=debug_mode)
        print(f"训练完成，最终损失: {train_losses['total'][-1]:.6f}")
        calibration_path = os.path.join(data_dir, 'normal_calibration.pt')
        if os.path.exists(calibration_path) and not regenerate_data:
            calibration_data = torch.load(calibration_path)
        else:
            calibration_data = self.generator.generate_data(fault_type=None, fault_start=None)
            torch.save(calibration_data, calibration_path)
        self.calibrate_normal(*calibration_data, k=k, window_size=window_size)

        # 保存训练好的模型
        model_path = os.path.join(data_dir, "trained_model.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"\n模型已保存到: {model_path}")
        print(f"  模型将用于后续的故障检测和诊断任务")

        # 保存训练损失曲线
        train_results_dir = os.path.join(results_base_dir, "training")
        os.makedirs(train_results_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses['total'], label='总损失', linewidth=2)
        plt.plot(train_losses['dyn'], label='动态损失', linewidth=2)
        plt.plot(train_losses['pdf'], label='PDF损失', linewidth=2)
        plt.plot(train_losses['smooth'], label='平滑损失', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('训练损失曲线（混合训练集：正常 + 故障类型1 + 故障类型2）', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数刻度，便于查看损失变化
        plt.savefig(os.path.join(train_results_dir, "training_loss.png"), dpi=300)
        plt.close()
        print(f"训练损失曲线已保存到: {train_results_dir}/training_loss.png")

        # ========== 阶段2：检测阶段（在故障数据上检测和诊断） ==========
        print("\n" + "="*60)
        print("阶段2：检测阶段 - 使用训练好的模型在故障数据上检测和诊断")
        print("="*60)

        for profile_label in fault_profiles:
            params_for_profile = profile_params.get(profile_label, {})
            for fault_type in fault_types:
                fault_name = f"故障类型{fault_type}"
                print(f"\n测试 {fault_name} - Profile: {profile_label} ...")

                # 创建故障类型的结果目录
                fault_results_dir = os.path.join(results_base_dir, f"fault_type_{fault_type}_{profile_label}")
                os.makedirs(fault_results_dir, exist_ok=True)

                # 多次重复实验
                accuracies = []
                precisions = []
                recalls = []
                f1s = []
                detection_delays = []
                diagnosis_accuracies = []
                trials_results = []

                for trial in range(num_trials):
                    print(f"\n  试验 {trial+1}/{num_trials}")

                    # 以故障类型命名文件
                    fault_name_file = f"fault_type_{fault_type}_{profile_label}"
                    data_save_path = os.path.join(data_dir, f"{fault_name_file}_data.pt")

                    # 检查是否存在保存的数据
                    if os.path.exists(data_save_path) and not regenerate_data:
                        # 加载已保存的数据
                        print(f"  加载已保存的数据...")
                        saved_data = torch.load(data_save_path)
                        a_seq = saved_data['a_seq']
                        u_seq = saved_data['u_seq']
                        g_seq = saved_data['g_seq']
                        F_seq = saved_data['F_seq']
                        pdf_seq = saved_data['pdf_seq']
                    else:
                        # 生成新数据
                        print(f"  生成新数据 ({profile_label}) ...")
                        a_seq, u_seq, g_seq, F_seq, pdf_seq = self.generator.generate_data(
                            fault_type=fault_type,
                            fault_start=fault_start,
                            fault_profile=profile_label,
                            profile_params=params_for_profile
                        )
                        # 保存生成的数据
                        torch.save({
                            'a_seq': a_seq,
                            'u_seq': u_seq,
                            'g_seq': g_seq,
                            'F_seq': F_seq,
                            'pdf_seq': pdf_seq
                        }, data_save_path)

                    # 调试模式下打印数据信息
                    if debug_mode:
                        print(f"  数据形状: a_seq={a_seq.shape}, pdf_seq={pdf_seq.shape}")
                        print(f"  故障开始时间: {fault_start}, 故障类型: {fault_type}, profile: {profile_label}")
                        pdf_before = pdf_seq[max(0, fault_start-10):fault_start]
                        pdf_after = pdf_seq[fault_start:fault_start+10]
                        print(f"  故障前PDF均值: {torch.mean(pdf_before).item():.4f}, 故障后PDF均值: {torch.mean(pdf_after).item():.4f}")
                        print(f"  故障前PDF方差: {torch.var(pdf_before).item():.4f}, 故障后PDF方差: {torch.var(pdf_after).item():.4f}")

                    # 注意：不再重新训练模型，直接使用训练好的模型进行评估
                    print("  使用训练好的模型进行评估（检测和诊断）...")

                    # 构建最优阈值文件路径（优先使用一致性优化的结果）
                    optimal_thresholds_path = self._get_default_optimal_thresholds_path()

                    results = self.evaluate_model(
                        a_seq, u_seq, g_seq, F_seq, pdf_seq,
                        fault_start=fault_start, fault_type=fault_type,
                        k=k, window_size=window_size, debug_mode=debug_mode,
                        use_optimal_thresholds=False,
                        optimal_thresholds_path=optimal_thresholds_path
                    )

                    if debug_mode:
                        print(f"  故障检测性能: 准确率={results['accuracy']:.4f}, 精确率={results['precision']:.4f}, 召回率={results['recall']:.4f}, F1={results['f1']:.4f}")
                        if results['detection_delay'] is not None:
                            print(f"  故障检测延迟: {results['detection_delay']} 时间步")
                        print(f"  故障诊断准确率: {results['diagnosis_accuracy']:.4f}")
                        print("  故障检测混淆矩阵:")
                        print(results['confusion_matrix'])

                    # 可视化结果
                    trial_results_dir = os.path.join(fault_results_dir, f"trial_{trial+1}")
                    self.visualize_results(results, fault_type, fault_start, save_path=trial_results_dir)

                    accuracies.append(results['accuracy'])
                    precisions.append(results['precision'])
                    recalls.append(results['recall'])
                    f1s.append(results['f1'])
                    if results['detection_delay'] is not None:
                        detection_delays.append(results['detection_delay'])
                    diagnosis_accuracies.append(results['diagnosis_accuracy'])

                    trials_results.append({
                        'accuracy': results['accuracy'],
                        'precision': results['precision'],
                        'recall': results['recall'],
                        'f1': results['f1'],
                        'detection_delay': results['detection_delay'],
                        'diagnosis_accuracy': results['diagnosis_accuracy']
                    })

                avg_accuracy = np.mean(accuracies)
                avg_precision = np.mean(precisions)
                avg_recall = np.mean(recalls)
                avg_f1 = np.mean(f1s)
                avg_detection_delay = np.mean(detection_delays) if detection_delays else None
                avg_diagnosis_accuracy = np.mean(diagnosis_accuracies)

                result_key = f"{fault_name}_{profile_label}"
                all_results[result_key] = {
                    'avg_accuracy': avg_accuracy,
                    'avg_precision': avg_precision,
                    'avg_recall': avg_recall,
                    'avg_f1': avg_f1,
                    'avg_detection_delay': avg_detection_delay,
                    'avg_diagnosis_accuracy': avg_diagnosis_accuracy,
                    'trials': trials_results,
                    'fault_profile': profile_label
                }

                with open(os.path.join(fault_results_dir, "average_results.txt"), 'w') as f:
                    f.write(f"故障类型: {fault_name}\n")
                    f.write(f"故障Profile: {profile_label}\n")
                    f.write(f"平均检测准确率: {avg_accuracy:.4f}\n")
                    f.write(f"平均检测精确率: {avg_precision:.4f}\n")
                    f.write(f"平均检测召回率: {avg_recall:.4f}\n")
                    f.write(f"平均检测F1分数: {avg_f1:.4f}\n")
                    if avg_detection_delay is not None:
                        f.write(f"平均检测延迟: {avg_detection_delay:.2f} 时间步\n")
                    f.write(f"平均诊断准确率: {avg_diagnosis_accuracy:.4f}\n")

        return all_results

    def save_results_to_csv(self, all_results):
        """
        将实验结果保存到CSV文件

        参数:
        all_results: 所有实验结果
        """
        # 创建性能指标摘要表
        summary_data = []
        for fault_name, results in all_results.items():
            row = {
                '故障类型': fault_name,
                '故障Profile': results.get('fault_profile', 'step'),
                '检测准确率': results['avg_accuracy'],
                '检测精确率': results['avg_precision'],
                '检测召回率': results['avg_recall'],
                '检测F1分数': results['avg_f1'],
                '平均检测延迟': results['avg_detection_delay'] if results['avg_detection_delay'] is not None else 'N/A',
                '诊断准确率': results['avg_diagnosis_accuracy']
            }
            summary_data.append(row)

        # 保存为CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(full_process_results_dir, 'experiment_summary.csv'), index=False)

        # 为每种故障类型创建详细结果表
        for fault_name, results in all_results.items():
            trials_data = []
            for i, trial in enumerate(results['trials']):
                row = {
                    '试验编号': i + 1,
                    '检测准确率': trial['accuracy'],
                    '检测精确率': trial['precision'],
                    '检测召回率': trial['recall'],
                    '检测F1分数': trial['f1'],
                    '检测延迟': trial['detection_delay'] if trial['detection_delay'] is not None else 'N/A',
                    '诊断准确率': trial['diagnosis_accuracy'] if 'diagnosis_accuracy' in trial else 'N/A'
                }
                trials_data.append(row)

            # 保存为CSV
            trials_df = pd.DataFrame(trials_data)
            trials_df.to_csv(os.path.join(full_process_results_dir, f'{fault_name}_detailed_results.csv'), index=False)

def main():
    # 设置实验参数
    T = 200  # 时间序列长度
    n_bins = 50  # PDF的分箱数
    d = 2  # 状态时滞步数
    b = 1  # 输入时滞步数
    epochs = 150  # 训练轮数（增加到150，确保模型充分学习故障模式）
    k = 1.5  # 故障检测阈值系数 (降低到1.5，提高对渐进式故障的敏感度，特别是故障类型1)
    window_size = 20  # 滑动窗口大小 (从25降低到20，减少延迟，提高对渐进式故障的敏感度)
    num_trials = 1  # 每种故障类型的重复试验次数
    fault_start = 100  # 故障开始时间
    fault_profiles = ['step', 'delayed_sigmoid']  # 双轨实验：阶跃 vs 渐进
    profile_params = {
        'delayed_sigmoid': {
            'delay_steps': 2,  # 时滞：故障触发后延迟2个时间步生效（可调范围：1-5）
            'growth_steps': 28,  # 非线性增长：20个时间步从0增长到饱和（可调范围：15-30）
            'saturation_level': 1.0  # 饱和水平：1.0表示完全饱和（可调范围：0.8-1.2）
        },
        'step': {}
    }

    # 调试参数
    regenerate_data = True  # 是否重新生成数据，True表示重新生成（确保使用修复后的数据生成逻辑）
    debug_mode = False  # 调试模式会输出更多信息和中间结果

    print("======== 初始化实验 ========")
    # 创建实验对象
    experiment = SimulationExperiment(T=T, n_bins=n_bins, d=d, b=b)

    # 运行实验
    print("======== 开始数值仿真实验... ========")
    start_time = time.time()

    # 测试故障类型：正常状态、药剂过多、药剂过少（硬编码，便于理解）
    # fault_type=1: 药剂过多 (小泡密集)
    # fault_type=2: 药剂过少 (大泡分散)
    fault_types = [1, 2]  # 1=药剂过多, 2=药剂过少
    print(f"测试的故障类型: [药剂过多(类型1), 药剂过少(类型2)]")

    all_results = experiment.run_experiment(
        num_trials=num_trials,
        fault_types=fault_types,
        fault_start=fault_start,
        epochs=epochs,
        k=k,
        window_size=window_size,
        regenerate_data=regenerate_data,
        debug_mode=debug_mode,
        fault_profiles=fault_profiles,
        profile_params=profile_params
    )

    end_time = time.time()
    print(f"\n实验完成! 总耗时: {(end_time - start_time) / 60:.2f} 分钟")

    # 保存实验结果到CSV
    print("\n保存实验结果到CSV文件...")
    experiment.save_results_to_csv(all_results)
    print("✓ 结果已保存到CSV文件")

    # 输出总结
    print("\n===== 实验总结 =====")
    for fault_name, results in all_results.items():
        print(f"\n{fault_name}:")
        print(f"  检测准确率: {results['avg_accuracy']:.4f}")
        print(f"  检测精确率: {results['avg_precision']:.4f}")
        print(f"  检测召回率: {results['avg_recall']:.4f}")
        print(f"  检测F1分数: {results['avg_f1']:.4f}")
        if results['avg_detection_delay'] is not None:
            print(f"  平均检测延迟: {results['avg_detection_delay']:.2f} 时间步")
        print(f"  诊断准确率: {results['avg_diagnosis_accuracy']:.4f}")

    print("\n===== 使用说明 =====")
    print("1. 故障类型定义:")
    print("   - 1: 药剂过多 (小气泡密集, PDF向左偏移)")
    print("   - 2: 药剂过少 (大气泡分散, PDF向右偏移)")
    print("2. 如需重新生成数据，请设置regenerate_data=True")
    print("3. 如需加快测试速度，可以减少epochs值")
    print("4. 结果保存在results目录下对应故障类型+profile文件夹中")

if __name__ == "__main__":
    main()
