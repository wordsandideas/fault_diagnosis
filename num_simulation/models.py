'''
浮选过程故障检测与诊断所需的模型组件
这是fusion_implementation.py的简化版本，只包含必要的类和函数，不加载外部数据
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# B样条基函数 PDF 估计模块
class BSplinePDF(nn.Module):
    def __init__(self, n_bins):
        """
        使用B样条的PDF生成网络

        参数:
        n_bins: PDF的分箱数
        """
        super(BSplinePDF, self).__init__()
        self.n_bins = n_bins
        self.n_basis = 10  # 使用10个B样条基函数

        # 创建控制点参数
        self.control_points = nn.Parameter(torch.rand(self.n_basis) * 0.5 + 0.25)

        # 创建神经网络来预测B样条控制点权重
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),  # 输入: 状态a, 非线性函数g, 故障F
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.n_basis)
        )

        # 初始化MLP权重，使用Xavier初始化
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, a, g, F):
        """
        前向传播，生成PDF

        参数:
        a: 状态值
        g: 非线性函数值
        F: 故障值

        返回:
        pdf: 生成的PDF
        """
        # 从输入预测B样条权重
        input_data = torch.cat([a, g, F], dim=0)
        weights = torch.softmax(self.mlp(input_data), dim=0)  # 使用softmax确保权重非负且和为1

        # 计算B样条PDF
        knots = torch.linspace(0, 1, self.n_bins, device=a.device)
        pdf = self._evaluate_bspline(knots, weights)

        # 归一化PDF以确保积分为1
        pdf = pdf / (torch.sum(pdf) * (1.0 / self.n_bins) + 1e-8)

        # 确保PDF值非负
        pdf = pdf.relu()

        # 平滑处理，防止过于尖锐的分布
        pdf = self._smooth_pdf(pdf, alpha=0.1)

        # 再次归一化
        pdf = pdf / (torch.sum(pdf) * (1.0 / self.n_bins) + 1e-8)

        return pdf

    def _evaluate_bspline(self, x, weights):
        """
        评估给定权重下的B样条函数

        参数:
        x: 评估点
        weights: B样条基函数权重

        返回:
        y: B样条函数值
        """
        # 创建均匀分布的节点
        t = torch.linspace(-0.2, 1.2, self.n_basis + 4, device=x.device)

        # 计算三次B样条基函数
        y = torch.zeros_like(x)
        for i in range(self.n_basis):
            # 计算每个点处的B样条基函数值
            basis = self._bspline_basis(x, t, i)
            # 加权求和
            y = y + weights[i] * basis

        return y

    def _bspline_basis(self, x, t, i, k=3):
        """
        计算B样条基函数

        参数:
        x: 评估点
        t: 节点序列
        i: 基函数索引
        k: 样条阶数 (默认为3，即三次样条)

        返回:
        basis: 基函数值
        """
        # 零阶B样条
        if k == 0:
            return ((x >= t[i]) & (x < t[i+1])).float()

        # 递归计算高阶B样条
        # 第一项
        term1 = torch.zeros_like(x)
        if t[i+k] - t[i] > 1e-8:
            term1 = (x - t[i]) / (t[i+k] - t[i]) * self._bspline_basis(x, t, i, k-1)

        # 第二项
        term2 = torch.zeros_like(x)
        if t[i+k+1] - t[i+1] > 1e-8:
            term2 = (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * self._bspline_basis(x, t, i+1, k-1)

        return term1 + term2

    def _smooth_pdf(self, pdf, alpha=0.1):
        """
        使用移动平均平滑PDF

        参数:
        pdf: 原始PDF
        alpha: 平滑参数

        返回:
        smoothed_pdf: 平滑后的PDF
        """
        kernel_size = 3
        padding = kernel_size // 2

        # 创建平滑核
        kernel = torch.ones(kernel_size, device=pdf.device) / kernel_size

        # 对原始PDF进行填充
        padded_pdf = F.pad(pdf.unsqueeze(0).unsqueeze(0), (padding, padding), mode='reflect')

        # 使用卷积进行平滑
        smoothed = F.conv1d(padded_pdf, kernel.view(1, 1, -1), padding=0)

        # 取出结果并混合原始PDF和平滑PDF
        smoothed_pdf = (1 - alpha) * pdf + alpha * smoothed.squeeze()

        return smoothed_pdf

    def smoothness_penalty(self):
        """
        计算PDF平滑度惩罚项

        返回:
        penalty: 平滑度惩罚值
        """
        # 计算控制点的二阶差分，作为平滑度度量
        diff1 = self.control_points[1:] - self.control_points[:-1]
        diff2 = diff1[1:] - diff1[:-1]

        # 计算平滑度惩罚项
        penalty = torch.mean(diff2 ** 2)

        return penalty

# 状态动力学模块（MLP）- 按照文章要求，预测动态权重系数 w = [A, A_d, H, H_d, G, J]
class StateDynamics(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=6):
        """
        状态动力学模型 - 预测公式(9)中的动态权重系数

        参数:
        input_dim: 输入维度 (a_td, u_tb, g_t, F_t) = 4
        hidden_dim: 隐藏层维度
        output_dim: 输出维度，默认为6 (对应 A, A_d, H, H_d, G, J 六个系数)
        """
        super(StateDynamics, self).__init__()
        self.hidden_dim = hidden_dim

        # 神经网络用来预测动态权值系数 A, A_d, H, H_d, G, J
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),  # 输出 6 个系数：A, A_d, H, H_d, G, J
        )

        # 系数偏置：避免初始系数为 0，使用较小的初始值
        self.coeff_bias = nn.Parameter(torch.tensor([-0.05, 0.02, 0.03, 0.01, 0.04, 0.0]), requires_grad=True)

        # 初始化权重，使用Xavier初始化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, a_t, a_td, u_t, u_tb, g_t, F_t):
        """
        前向传播 - 预测动态权重系数，然后根据公式(9)计算状态变化率

        参数:
        a_t: 当前状态 a(t)
        a_td: 时滞d的状态 a(t-d)
        u_t: 当前输入 u(t)
        u_tb: 时滞b的输入 u(t-b)
        g_t: 当前的非线性函数 g(t)
        F_t: 当前的故障 F(t)

        返回:
        da_dt: 状态变化率（根据公式(9)计算）
        coefficients: 预测的系数 [A, A_d, H, H_d, G, J]
        """
        # 输入特征：a(t-d), u(t-b), g(t), F(t)
        # 确保所有输入都是标量或1D张量
        a_td = a_td.squeeze() if a_td.dim() > 0 else a_td
        u_tb = u_tb.squeeze() if u_tb.dim() > 0 else u_tb
        g_t = g_t.squeeze() if g_t.dim() > 0 else g_t
        F_t = F_t.squeeze() if F_t.dim() > 0 else F_t

        input_data = torch.stack([a_td, u_tb, g_t, F_t]).unsqueeze(0)  # [1, 4]

        # 通过网络预测动态权值系数 A, A_d, H, H_d, G, J
        coefficients = self.net(input_data).squeeze(0) + self.coeff_bias  # [6]

        # 约束系数范围，防止过大（使用tanh约束到合理范围）
        # 但不要过度约束，允许系数有合理的范围
        coefficients = torch.tanh(coefficients) * 0.15  # 限制系数在 [-0.15, 0.15] 范围内，稍微放宽

        # 根据公式(9)计算状态变化率 da(t)/dt
        # da/dt = A * a(t) + A_d * a(t-d) + H * u(t) + H_d * u(t-b) + G * g(t) + J * F(t)
        da_dt = (
            coefficients[0] * a_t +      # A * a(t)
            coefficients[1] * a_td +    # A_d * a(t-d)
            coefficients[2] * u_t +      # H * u(t)
            coefficients[3] * u_tb +    # H_d * u(t-b)
            coefficients[4] * g_t +     # G * g(t)
            coefficients[5] * F_t       # J * F(t)
        )

        # 约束状态变化率，防止过大（但不要过度约束，允许合理的变化）
        da_dt = torch.clamp(da_dt, -2.0, 2.0)  # 稍微放宽限制

        # 确保da_dt是标量
        if da_dt.dim() > 0:
            da_dt = da_dt.squeeze()

        return da_dt, coefficients

# 整体PINNs模型结构--主模型，把前两个模块整合到一起
class PINN_BSpline_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_bins, d=2, b=1):
        """
        基于物理信息的神经网络模型，集成了B样条PDF网络和状态动力学

        参数:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        n_bins: PDF的分箱数
        d: 状态时滞步数
        b: 输入时滞步数
        """
        super(PINN_BSpline_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_bins = n_bins
        self.d = d
        self.b = b

        # 两个子模型
        self.pdf_net = BSplinePDF(n_bins)
        # StateDynamics预测6个系数（A, A_d, H, H_d, G, J），然后根据公式(9)计算状态变化率
        self.state_net = StateDynamics(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=6)

    def forward(self, a_seq, u_seq, g_seq, F_seq):
        """
        前向传播

        参数:
        a_seq: 状态序列 (T, 1)
        u_seq: 控制输入序列 (T, 1)
        g_seq: 非线性函数序列 (T, 1)
        F_seq: 故障序列 (T, 1)

        返回:
        pred_a_seq: 预测的状态序列 (T-1, 1)
        pred_pdf_seq: 预测的PDF序列 (T, n_bins)
        """
        T = len(a_seq)
        pred_a_seq = torch.zeros(T-1, 1, device=a_seq.device)
        pred_pdf_seq = torch.zeros(T, self.n_bins, device=a_seq.device)

        # 计算每个时间步的PDF
        for t in range(T):
            pred_pdf_seq[t] = self.pdf_net(a_seq[t], g_seq[t], F_seq[t])

        # 计算每个时间步的状态 - 按照文章要求，使用公式(9)
        for t in range(max(self.d, self.b), T-1):
            # 收集输入特征: a(t), a(t-d), u(t), u(t-b), g(t), F(t)
            a_t = a_seq[t]  # 当前状态 a(t)
            a_td = a_seq[t-self.d]  # 时滞d的状态 a(t-d)
            u_t = u_seq[t]  # 当前输入 u(t)
            u_tb = u_seq[t-self.b]  # 时滞b的输入 u(t-b)
            g_t = g_seq[t]  # 当前的非线性函数 g(t)
            F_t = F_seq[t]  # 当前的故障 F(t)

            # 使用StateDynamics预测动态权重系数，然后根据公式(9)计算状态变化率
            da_dt, coefficients = self.state_net(a_t, a_td, u_t, u_tb, g_t, F_t)

            # 根据公式(9): x(k+1) = x(k) + da_dt，预测下一个状态
            pred_a_seq[t] = a_t + da_dt

        # 初始化区间保留占位值；训练和评分必须排除该区间。

        return pred_a_seq, pred_pdf_seq

# 检测故障 -- 使用滑动窗口对残差进行动态阈值判断
def detect_fault(a_pred_seq, a_true_seq, k, window_size):   # window_size: 滑动窗口大小（用于估计正常残差的统计量）
    # 计算残差并取绝对值
    residuals = torch.abs(a_pred_seq - a_true_seq[:-1])

    # 处理 NaN 值：使用前向填充
    if torch.isnan(residuals).any():
        for i in range(len(residuals)):
            if torch.isnan(residuals[i]):
                if i > 0:
                    residuals[i] = residuals[i-1]
                else:
                    residuals[i] = torch.tensor(0.0, device=residuals.device)

    # 初始化故障标志和阈值
    fault_flags = torch.zeros_like(residuals)
    thresholds = torch.zeros_like(residuals)

    # 计算残差的滑动均值和标准差
    moving_mean = torch.zeros_like(residuals)
    moving_std = torch.zeros_like(residuals)

    # 首先计算初始窗口的统计量
    if len(residuals) > window_size:
        base_window = residuals[:window_size]
        # 过滤 NaN 值（虽然已经处理过，但为了安全再次过滤）
        valid_base_window = base_window[~torch.isnan(base_window)]
        if len(valid_base_window) > 0:
            base_mean = torch.mean(valid_base_window)
            base_std = torch.std(valid_base_window)
        else:
            # 如果没有有效值，使用默认值
            base_mean = torch.tensor(0.0, device=residuals.device)
            base_std = torch.tensor(1.0, device=residuals.device)

        # 初始化前window_size个阈值
        thresholds[:window_size] = base_mean + k * base_std
        moving_mean[:window_size] = base_mean
        moving_std[:window_size] = base_std
    else:
        # 如果序列长度小于窗口大小，使用全部数据
        valid_residuals = residuals[~torch.isnan(residuals)]
        if len(valid_residuals) > 0:
            base_mean = torch.mean(valid_residuals)
            base_std = torch.std(valid_residuals)
        else:
            base_mean = torch.tensor(0.0, device=residuals.device)
            base_std = torch.tensor(1.0, device=residuals.device)
        thresholds[:] = base_mean + k * base_std
        moving_mean[:] = base_mean
        moving_std[:] = base_std

    # 使用CUSUM算法来检测变化点
    cusum_pos = torch.zeros_like(residuals)
    cusum_neg = torch.zeros_like(residuals)
    drift = 0.5  # 漂移参数
    threshold_cusum = 5.0  # CUSUM阈值

    # 对每个时间点进行处理
    for t in range(window_size, len(residuals)):
        # 计算当前窗口的统计量
        window_vals = residuals[max(0, t - window_size):t]
        # 过滤 NaN 值
        valid_window_vals = window_vals[~torch.isnan(window_vals)]
        if len(valid_window_vals) > 0:
            current_mean = torch.mean(valid_window_vals)
            current_std = torch.std(valid_window_vals)
        else:
            # 如果没有有效值，使用前一个统计量
            current_mean = moving_mean[t-1] if t > 0 else base_mean
            current_std = moving_std[t-1] if t > 0 else base_std

        # 使用指数加权平均进行平滑
        alpha = 0.3  # 平滑因子，增大以更快响应变化
        if t > window_size:
            moving_mean[t] = alpha * current_mean + (1 - alpha) * moving_mean[t-1]
            moving_std[t] = alpha * current_std + (1 - alpha) * moving_std[t-1]
        else:
            moving_mean[t] = current_mean
            moving_std[t] = current_std

        # 计算自适应阈值，使用较大的k值增强检测灵敏度
        adaptive_k = k * (1.0 + 0.5 * torch.sigmoid(residuals[t] - moving_mean[t] - 2 * moving_std[t]))
        thresholds[t] = moving_mean[t] + adaptive_k * moving_std[t]

        # 更新CUSUM统计量
        normalized_residual = (residuals[t] - moving_mean[t]) / (moving_std[t] + 1e-6)
        cusum_pos[t] = torch.max(torch.tensor(0.0).to(residuals.device), cusum_pos[t-1] + normalized_residual - drift)
        cusum_neg[t] = torch.max(torch.tensor(0.0).to(residuals.device), cusum_neg[t-1] - normalized_residual - drift)

        # 多条件故障检测
        # 1. 基于阈值的检测
        if residuals[t] > thresholds[t]:
            fault_flags[t] = 1.0
        # 2. 基于CUSUM的检测
        elif cusum_pos[t] > threshold_cusum or cusum_neg[t] > threshold_cusum:
            fault_flags[t] = 1.0
        # 3. 连续超过较低阈值的检测
        elif t > 1 and residuals[t] > 0.8 * thresholds[t] and residuals[t-1] > 0.8 * thresholds[t-1] and residuals[t-2] > 0.8 * thresholds[t-2]:
            fault_flags[t] = 1.0
        # 4. 突变检测
        elif t > 0 and residuals[t] > residuals[t-1] * 2.0 and residuals[t] > 0.7 * thresholds[t]:
            fault_flags[t] = 1.0

        # 应用滞后效应：如果检测到故障，后续几个时间步也标记为故障
        if t > 0 and fault_flags[t-1] == 1.0:
            fault_persistence = 5  # 增加故障持续时间
            for i in range(1, fault_persistence + 1):
                if t + i < len(fault_flags):
                    fault_flags[t + i] = 1.0

    # 平滑故障标志，减少误报
    for t in range(1, len(fault_flags) - 1):
        # 如果前后都是故障，中间孤立的非故障点也标记为故障
        if fault_flags[t-1] == 1.0 and fault_flags[t+1] == 1.0 and fault_flags[t] == 0.0:
            fault_flags[t] = 1.0

    return fault_flags, residuals, thresholds

# 提取无故障状态下的特征均值和标准差，用于诊断故障状态下的具体故障
def compute_normal_stats(pdf_pred_seq, fault_flags):
    """
    计算正常状态下的PDF特征统计量

    参数:
    pdf_pred_seq: 预测的PDF序列
    fault_flags: 故障标志

    返回:
    stats: 正常状态下的特征统计量
    """
    x_bins = torch.linspace(0, 1, pdf_pred_seq.size(1))  # 泡沫粒径的x轴
    stats = {
        'peak': [], 'mean': [], 'std': [], 'skew': [],
        'kurtosis': [], 'q25': [], 'q50': [], 'q75': [], 'mode': []
    }
    fault_flags = torch.any(fault_flags != 0, dim=1)  # [T-1]，每个窗口是否有故障

    for pdf, flag in zip(pdf_pred_seq, fault_flags):
        if not flag:  # 只提取无故障样本
            pdf = pdf / (torch.sum(pdf) + 1e-8)
            pdf_np = pdf.detach().cpu().numpy()
            x_np = x_bins.detach().cpu().numpy()

            # 基本统计量
            peak = np.max(pdf_np)
            mean = np.sum(pdf_np * x_np)
            std_dev = np.sqrt(np.sum(pdf_np * (x_np - mean) ** 2))
            skewness = np.sum(((x_np - mean) ** 3) * pdf_np) / (std_dev ** 3 + 1e-8)
            kurtosis_val = np.sum(((x_np - mean) ** 4) * pdf_np) / (std_dev ** 4 + 1e-8) - 3  # 超峰度

            # 分位数
            cdf = np.cumsum(pdf_np)
            cdf = cdf / cdf[-1]  # 归一化CDF
            q25 = np.interp(0.25, cdf, x_np)
            q50 = np.interp(0.50, cdf, x_np)
            q75 = np.interp(0.75, cdf, x_np)

            # 众数（最大概率密度对应的x值）
            mode_idx = np.argmax(pdf_np)
            mode = x_np[mode_idx]

            # 保存统计量
            stats['peak'].append(peak)
            stats['mean'].append(mean)
            stats['std'].append(std_dev)
            stats['skew'].append(skewness)
            stats['kurtosis'].append(kurtosis_val)
            stats['q25'].append(q25)
            stats['q50'].append(q50)
            stats['q75'].append(q75)
            stats['mode'].append(mode)

    # 如果没有无故障样本，使用全部样本的前20%作为替代
    if not stats['peak']:
        n_samples = len(pdf_pred_seq)
        n_normal = max(5, int(n_samples * 0.2))  # 至少5个样本

        for t in range(n_normal):
            pdf = pdf_pred_seq[t]
            pdf = pdf / (torch.sum(pdf) + 1e-8)
            pdf_np = pdf.detach().cpu().numpy()
            x_np = x_bins.detach().cpu().numpy()

            peak = np.max(pdf_np)
            mean = np.sum(pdf_np * x_np)
            std_dev = np.sqrt(np.sum(pdf_np * (x_np - mean) ** 2))
            skewness = np.sum(((x_np - mean) ** 3) * pdf_np) / (std_dev ** 3 + 1e-8)
            kurtosis_val = np.sum(((x_np - mean) ** 4) * pdf_np) / (std_dev ** 4 + 1e-8) - 3

            # 分位数
            cdf = np.cumsum(pdf_np)
            cdf = cdf / cdf[-1]
            q25 = np.interp(0.25, cdf, x_np)
            q50 = np.interp(0.50, cdf, x_np)
            q75 = np.interp(0.75, cdf, x_np)

            # 众数
            mode_idx = np.argmax(pdf_np)
            mode = x_np[mode_idx]

            stats['peak'].append(peak)
            stats['mean'].append(mean)
            stats['std'].append(std_dev)
            stats['skew'].append(skewness)
            stats['kurtosis'].append(kurtosis_val)
            stats['q25'].append(q25)
            stats['q50'].append(q50)
            stats['q75'].append(q75)
            stats['mode'].append(mode)

    # 返回均值和标准差
    return {
        k: (np.mean(v), np.std(v))
        for k, v in stats.items()
    }

def diagnose_fault(pdf_pred_seq, fault_flags, normal_stats, visualize=True, std_factor=1.5):
    """
    诊断故障类型

    参数:
    pdf_pred_seq: 预测的PDF序列
    fault_flags: 故障标志
    normal_stats: 正常状态下的特征统计量
    visualize: 是否可视化
    std_factor: 标准差因子，用于判断异常

    返回:
    diagnoses: 诊断结果列表
    """
    diagnoses = []
    x_bins = torch.linspace(0, 1, pdf_pred_seq.size(1))
    fault_flags = torch.any(fault_flags != 0, dim=1)  # [T-1]，每个窗口是否有故障

    # 如果需要可视化，导入matplotlib
    if visualize:
        import matplotlib.pyplot as plt

    # 提取正常统计量
    μ_peak, σ_peak = normal_stats['peak']
    μ_mean, σ_mean = normal_stats['mean']
    μ_std, σ_std = normal_stats['std']
    μ_skew, σ_skew = normal_stats['skew']
    μ_kurtosis, σ_kurtosis = normal_stats['kurtosis']
    μ_q25, σ_q25 = normal_stats['q25']
    μ_q50, σ_q50 = normal_stats['q50']
    μ_q75, σ_q75 = normal_stats['q75']
    μ_mode, σ_mode = normal_stats['mode']

    # 计算四分位距
    μ_iqr = μ_q75 - μ_q25

    # 增强的故障模式特征向量 - 更精确的特征模式定义
    fault_patterns = {
        "小泡密集，浮选药剂过少": np.array([2.5, -2.0, -1.5, 2.0, 2.5, -2.0, -1.5]),
        "大泡分散，浮选药剂过多": np.array([-1.5, 1.5, 2.0, -1.5, -2.0, 1.5, 2.0]),
        "偏左：药剂供给不足": np.array([1.0, -2.5, -0.5, 1.0, 0.5, -2.0, -0.5]),
        "偏右：药剂浓度偏高": np.array([1.0, 2.5, 0.5, -1.0, -0.5, 2.0, 0.5]),
        "泡径极度集中，可能药剂局部过饱和": np.array([3.0, 0.0, -3.0, 0.0, 4.0, 0.0, -3.0])
    }

    # 定义特征权重，突出重要特征
    feature_weights = np.array([1.0, 2.0, 1.5, 1.2, 1.0, 1.8, 1.5])

    # 存储连续诊断结果，用于平滑
    diagnosis_history = []

    # 若某一时间步t被标记为存在故障，就提取PDF特征并诊断
    for t, (pdf, fault_flag) in enumerate(zip(pdf_pred_seq, fault_flags)):
        if fault_flag:
            pdf = pdf / (torch.sum(pdf) + 1e-8)
            pdf_np = pdf.detach().cpu().numpy()
            x_np = x_bins.detach().cpu().numpy()

            # 特征提取
            peak = np.max(pdf_np)
            mean = np.sum(pdf_np * x_np)
            std_dev = np.sqrt(np.sum(pdf_np * (x_np - mean) ** 2))
            skewness = np.sum(((x_np - mean) ** 3) * pdf_np) / (std_dev ** 3 + 1e-8)
            kurtosis_val = np.sum(((x_np - mean) ** 4) * pdf_np) / (std_dev ** 4 + 1e-8) - 3

            # 分位数
            cdf = np.cumsum(pdf_np)
            cdf = cdf / (cdf[-1] + 1e-8)  # 归一化CDF
            q25 = np.interp(0.25, cdf, x_np)
            q50 = np.interp(0.50, cdf, x_np)
            q75 = np.interp(0.75, cdf, x_np)
            iqr = q75 - q25

            # 众数
            mode_idx = np.argmax(pdf_np)
            mode = x_np[mode_idx]

            # 计算特征偏离程度
            peak_dev = (peak - μ_peak) / (σ_peak + 1e-8)
            mean_dev = (mean - μ_mean) / (σ_mean + 1e-8)
            std_dev_rel = (std_dev - μ_std) / (σ_std + 1e-8)
            skew_dev = (skewness - μ_skew) / (σ_skew + 1e-8)
            kurt_dev = (kurtosis_val - μ_kurtosis) / (σ_kurtosis + 1e-8)
            mode_dev = (mode - μ_mode) / (σ_mode + 1e-8)
            iqr_dev = (iqr - μ_iqr) / (σ_q75 + σ_q25 + 1e-8)

            # 构建特征向量并应用权重
            feature_vector = np.array([
                peak_dev * feature_weights[0],
                mean_dev * feature_weights[1],
                std_dev_rel * feature_weights[2],
                skew_dev * feature_weights[3],
                kurt_dev * feature_weights[4],
                mode_dev * feature_weights[5],
                iqr_dev * feature_weights[6]
            ])

            # 计算与各故障模式的相似度（加权余弦相似度）
            similarities = {}
            for fault_name, pattern in fault_patterns.items():
                # 计算余弦相似度
                weighted_pattern = np.array([
                    pattern[0] * feature_weights[0],
                    pattern[1] * feature_weights[1],
                    pattern[2] * feature_weights[2],
                    pattern[3] * feature_weights[3],
                    pattern[4] * feature_weights[4],
                    pattern[5] * feature_weights[5],
                    pattern[6] * feature_weights[6]
                ])
                similarity = np.dot(feature_vector, weighted_pattern) / (np.linalg.norm(feature_vector) * np.linalg.norm(weighted_pattern) + 1e-8)
                similarities[fault_name] = similarity

            # 找出最相似的故障类型
            best_match = max(similarities.items(), key=lambda x: x[1])
            result = best_match[0]
            confidence = best_match[1]

            # 如果相似度太低，使用更严格的规则诊断
            if confidence < 0.4:
                # 基于多个特征的综合判断
                if peak > μ_peak + 2.0 * σ_peak and mean < μ_mean - 1.0 * σ_mean and std_dev < μ_std:
                    result = "小泡密集，浮选药剂过少"
                    confidence = 0.7
                elif std_dev > μ_std + 1.5 * σ_std and mean > μ_mean + 1.0 * σ_mean:
                    result = "大泡分散，浮选药剂过多"
                    confidence = 0.7
                elif mean < μ_mean - 1.8 * σ_mean and mode < μ_mode - 1.5 * σ_mode:
                    result = "偏左：药剂供给不足"
                    confidence = 0.7
                elif mean > μ_mean + 1.8 * σ_mean and mode > μ_mode + 1.5 * σ_mode:
                    result = "偏右：药剂浓度偏高"
                    confidence = 0.7
                elif std_dev < μ_std - 1.8 * σ_std and kurtosis_val > μ_kurtosis + 2.0 * σ_kurtosis:
                    result = "泡径极度集中，可能药剂局部过饱和"
                    confidence = 0.7
                else:
                    # 使用次优匹配作为备选
                    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_matches) > 1 and sorted_matches[1][1] > 0.2:
                        result = sorted_matches[1][0]
                        confidence = sorted_matches[1][1]
                    else:
                        result = "未知故障模式"
                        confidence = 0.1

            # 应用诊断结果平滑
            diagnosis_history.append((result, confidence))
            if len(diagnosis_history) > 3:
                # 如果连续3个诊断结果相同，增强置信度
                recent_results = [d[0] for d in diagnosis_history[-3:]]
                if all(r == result for r in recent_results):
                    confidence = min(1.0, float(confidence + 0.2))

                # 如果当前诊断与历史不符但置信度低，可能采用历史诊断
                elif confidence < 0.5:
                    # 查找历史中出现频率最高的诊断
                    from collections import Counter
                    result_counts = Counter([d[0] for d in diagnosis_history[-5:]])
                    most_common = result_counts.most_common(1)[0]
                    if most_common[1] >= 3 and most_common[0] != result:
                        result = most_common[0]
                        confidence = 0.6

            # 添加诊断结果
            diagnoses.append((t, peak, std_dev, skewness, result, confidence))

            # 画出每个故障时刻的预测PDF曲线
            if visualize:
                plt.figure(figsize=(10, 6))
                plt.plot(x_np, pdf_np, 'b-', linewidth=2, label=f"t={t}")

                # 添加特征标记
                plt.axvline(x=float(mean), color='r', linestyle='--', label=f'均值: {mean:.3f}')
                plt.axvline(x=float(mode), color='g', linestyle=':', label=f'众数: {mode:.3f}')
                plt.axvline(x=float(q25), color='m', linestyle='-.', label=f'Q25: {q25:.3f}')
                plt.axvline(x=float(q75), color='y', linestyle='-.', label=f'Q75: {q75:.3f}')

                plt.title(f"故障时刻 t={t} 的PDF曲线\n诊断: {result} (置信度: {confidence:.2f})")
                plt.xlabel("归一化泡径")
                plt.ylabel("概率密度")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    # 对诊断结果进行后处理，解决孤立的不一致诊断
    if len(diagnoses) > 2:
        processed_diagnoses = []
        for i, diag in enumerate(diagnoses):
            t, peak, std_dev, skewness, result, confidence = diag

            # 检查前后诊断结果是否一致
            if i > 0 and i < len(diagnoses) - 1:
                prev_result = diagnoses[i-1][4]
                next_result = diagnoses[i+1][4]

                # 如果当前诊断与前后不一致但前后一致，则采用前后的诊断结果
                if prev_result == next_result and result != prev_result and confidence < 0.7:
                    result = prev_result
                    confidence = 0.6

            processed_diagnoses.append((t, peak, std_dev, skewness, result, confidence))

        return processed_diagnoses

    return diagnoses
