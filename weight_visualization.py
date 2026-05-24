import mynn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ======================== config ========================
MODEL_PATH = r'./best_models_cnn/best_model.pickle'   # 修改为你的模型路径
MODEL_TYPE = 'cnn'                                     # 'mlp' 或 'cnn'
SAVE_DIR = r'./visualizations'                        # 图片保存目录
# ========================================================

import os
os.makedirs(SAVE_DIR, exist_ok=True)


def detect_model_type(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    if isinstance(raw[0], list) and len(raw[0]) > 0 and isinstance(raw[0][0], dict):
        return 'cnn'
    return 'mlp'


MODEL_TYPE = detect_model_type(MODEL_PATH)
print(f"Detected model type: {MODEL_TYPE}")

# load model
if MODEL_TYPE == 'mlp':
    model = nn.models.Model_MLP()
elif MODEL_TYPE == 'cnn':
    model = nn.models.Model_CNN()
model.load_model(MODEL_PATH)


def visualize_mlp_weights(model, save_dir):
    """
    MLP 第一层权重可视化。
    W shape: [784, 256]，每列是一个神经元的权重，reshape 为 28x28。
    好的结果：能看到模糊的数字笔画、边缘检测器、光晕等结构化模式。
    差的结果（需警惕）：纯噪声、无规则斑点 — 说明模型欠训练或学习率不当。
    """
    # 找到第一个 Linear 层
    linear_layer = None
    for layer in model.layers:
        if isinstance(layer, nn.op.Linear):
            linear_layer = layer
            break

    W = linear_layer.params['W']  # [784, 256]
    print(f"MLP Layer 1 weight shape: {W.shape}")

    n_neurons = W.shape[1]  # 256
    cols = 16
    rows = (n_neurons + cols - 1) // cols  # 16 x 16 = 256

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    fig.suptitle('MLP First Layer Weights (each 28x28)', fontsize=14)

    vmin, vmax = W.min(), W.max()
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if i < n_neurons:
            w_img = W[:, i].reshape(28, 28)
            ax.imshow(w_img, cmap='RdBu', vmin=vmin, vmax=vmax)
        else:
            ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'mlp_layer1_weights.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def visualize_mlp_weights_highlight(model, save_dir, n_highlight=16):
    """
    选取权重范数最大的前 n_highlight 个神经元重点展示，
    并标注其 L2 范数（范数越大 = 该神经元对输入越敏感）。
    """
    linear_layer = None
    for layer in model.layers:
        if isinstance(layer, nn.op.Linear):
            linear_layer = layer
            break

    W = linear_layer.params['W']  # [784, 256]
    norms = np.linalg.norm(W, axis=0)  # L2 norm per neuron
    top_idx = np.argsort(norms)[-n_highlight:][::-1]

    cols = 8
    rows = n_highlight // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f'Top {n_highlight} MLP Neurons (by L2 norm)', fontsize=14)

    for j, idx in enumerate(top_idx):
        ax = axes[j // cols][j % cols]
        w_img = W[:, idx].reshape(28, 28)
        ax.imshow(w_img, cmap='gray')
        ax.set_title(f'#{idx} |{norms[idx]:.2f}|', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'mlp_top_neurons.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def visualize_cnn_kernels(model, save_dir):
    """
    新 CNN 架构权重可视化:
      conv1: 32 filters, 1×3×3 → 可直接显示每个 3×3 kernel
      conv2: 64 filters, 32×3×3 → 取绝对值平均展示综合模式
      fc1:   [3136, 128] → 按 L2 范数挑 top-16 神经元
      fc2:   [128, 10] → 权重矩阵热力图
    """
    conv_layers = []
    linear_layers = []
    for layer in model.layers:
        if isinstance(layer, nn.op.conv2D):
            conv_layers.append(layer)
        if isinstance(layer, nn.op.Linear):
            linear_layers.append(layer)

    # ---- conv1: 32 filters, 1 input channel, 3×3 ----
    if conv_layers:
        W = conv_layers[0].params['W']  # [32, 1, 3, 3]
        out_c, in_c, K, _ = W.shape
        print(f"Conv1 weight shape: {W.shape}")

        cols, rows = 8, 4
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        fig.suptitle(f'Conv1 Kernels ({out_c} filters, {K}×{K})', fontsize=14)
        vmin, vmax = W.min(), W.max()
        for i in range(out_c):
            ax = axes[i // cols][i % cols]
            ax.imshow(W[i, 0], cmap='RdBu', vmin=vmin, vmax=vmax)
            ax.set_title(f'F{i}', fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        for i in range(out_c, rows * cols):
            axes[i // cols][i % cols].axis('off')
        plt.tight_layout()
        p = os.path.join(save_dir, 'cnn_conv1_kernels.png')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close(fig)

    # ---- conv2: 64 filters, 32 input channels, 3×3 (avg over in-channels) ----
    if len(conv_layers) >= 2:
        W = conv_layers[1].params['W']  # [64, 32, 3, 3]
        out_c, in_c, K, _ = W.shape
        print(f"Conv2 weight shape: {W.shape}")

        cols, rows = 8, 8
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        fig.suptitle(f'Conv2 Kernels ({out_c} filters, {K}×{K}, avg over {in_c} in-ch)', fontsize=14)
        for i in range(out_c):
            ax = axes[i // cols][i % cols]
            avg_kernel = np.mean(np.abs(W[i]), axis=0)
            ax.imshow(avg_kernel, cmap='gray')
            ax.set_title(f'F{i}', fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
        for i in range(out_c, rows * cols):
            axes[i // cols][i % cols].axis('off')
        plt.tight_layout()
        p = os.path.join(save_dir, 'cnn_conv2_kernels.png')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close(fig)

        # 挑一个 filter 展示其全部 32 个输入通道的 kernel
        for fi in range(min(3, out_c)):
            in_cols, in_rows = 8, 4
            fig, axes = plt.subplots(in_rows, in_cols, figsize=(in_cols * 1.2, in_rows * 1.2))
            fig.suptitle(f'Conv2 Filter {fi}: {in_c} input-channel kernels (3×3)', fontsize=11)
            vmin, vmax = W[fi].min(), W[fi].max()
            for j in range(in_c):
                ax = axes[j // in_cols][j % in_cols]
                ax.imshow(W[fi, j], cmap='RdBu', vmin=vmin, vmax=vmax)
                ax.set_title(f'ch{j}', fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
            for j in range(in_c, in_rows * in_cols):
                axes[j // in_cols][j % in_cols].axis('off')
            plt.tight_layout()
            p = os.path.join(save_dir, f'cnn_conv2_filter{fi}_detail.png')
            fig.savefig(p, dpi=200, bbox_inches='tight')
            print(f"Saved: {p}")
            plt.close(fig)

    # ---- fc1: [3136, 128] → top-16 neurons by L2 norm ----
    if linear_layers:
        W = linear_layers[0].params['W']  # [3136, 128]
        print(f"FC1 weight shape: {W.shape}")
        norms = np.linalg.norm(W, axis=0)
        top_idx = np.argsort(norms)[-16:][::-1]

        cols, rows = 8, 2
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        fig.suptitle('FC1 Top-16 Neurons (by L2 norm)', fontsize=14)
        vmin, vmax = W.min(), W.max()
        for j, idx in enumerate(top_idx):
            ax = axes[j // cols][j % cols]
            w_img = W[:, idx].reshape(64, 7, 7)
            # sum over channel dim → spatial heatmap
            spatial = np.mean(np.abs(w_img), axis=0)
            ax.imshow(spatial, cmap='hot')
            ax.set_title(f'#{idx} |{norms[idx]:.1f}|', fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        p = os.path.join(save_dir, 'cnn_fc1_top_neurons.png')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close(fig)

    # ---- fc2: [128, 10] → weight matrix heatmap ----
    if len(linear_layers) >= 2:
        W = linear_layers[1].params['W']  # [128, 10]
        print(f"FC2 weight shape: {W.shape}")
        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(W, cmap='RdBu', aspect='auto')
        ax.set_title('FC2 Weight Matrix (128→10)')
        ax.set_xlabel('Output class')
        ax.set_ylabel('Hidden neuron')
        ax.set_xticks(range(10))
        ax.set_yticks(range(0, 128, 16))
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        p = os.path.join(save_dir, 'cnn_fc2_weight_matrix.png')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close(fig)


# ======================== run ========================
print(f"Model type: {MODEL_TYPE}")
print(f"Model path: {MODEL_PATH}")
print("-" * 50)

if MODEL_TYPE == 'mlp':
    visualize_mlp_weights(model, SAVE_DIR)
    visualize_mlp_weights_highlight(model, SAVE_DIR)
elif MODEL_TYPE == 'cnn':
    visualize_cnn_kernels(model, SAVE_DIR)

print("-" * 50)
print("Done. All visualizations saved to:", SAVE_DIR)
