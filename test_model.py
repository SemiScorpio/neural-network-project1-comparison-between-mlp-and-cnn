import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=r'./best_models_cnn/best_model.pickle',
                    help='Path to saved model pickle')
parser.add_argument('--robustness', action='store_true', default=True,
                    help='Run robustness analysis')
parser.add_argument('--plot_confusion', action='store_true', default=True,
                    help='Plot and save confusion matrix')
parser.add_argument('--save_dir', type=str, default=r'./visualizations',
                    help='Directory for output figures')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, args.save_dir), exist_ok=True)

# ---------------- load test set ----------------
test_images_path = os.path.join(script_dir, 'dataset', 'MNIST', 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(script_dir, 'dataset', 'MNIST', 't10k-labels-idx1-ubyte.gz')

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs.astype(np.float64) / 255.0

if not os.path.exists(args.model_path):
    args.model_path = os.path.join(script_dir, args.model_path)

# ---------------- detect model type ----------------
with open(args.model_path, 'rb') as f:
    raw = pickle.load(f)

if isinstance(raw[0], list) and len(raw[0]) > 0 and isinstance(raw[0][0], dict):
    model_type = 'cnn'
    model = nn.models.Model_CNN()
else:
    model_type = 'mlp'
    model = nn.models.Model_MLP()

print(f"Model: {model_type.upper()}")
print(f"Path: {args.model_path}")

model.load_model(args.model_path)
model.set_training(False)

# ---------------- prepare input ----------------
if model_type == 'cnn':
    test_imgs_4d = test_imgs.reshape(-1, 1, 28, 28)
else:
    test_imgs_4d = test_imgs


def evaluate(imgs, labs):
    if model_type == 'cnn' and imgs.ndim == 2:
        imgs = imgs.reshape(-1, 1, 28, 28)
    elif model_type == 'mlp' and imgs.ndim == 4:
        imgs = imgs.reshape(imgs.shape[0], -1)
    logits = model(imgs)
    return nn.metric.accuracy(logits, labs)


# ---------------- clean evaluation ----------------
clean_acc = evaluate(test_imgs_4d, test_labs)
preds = np.argmax(model(test_imgs_4d if model_type == 'cnn' else test_imgs), axis=1)

print(f"\nClean Test Accuracy: {clean_acc:.4f} ({clean_acc * 100:.2f}%)")

# per-class accuracy
per_class = {}
for cls in range(10):
    mask = test_labs == cls
    per_class[cls] = (preds[mask] == cls).sum() / mask.sum()
    print(f"  Digit {cls}: {per_class[cls]:.4f}")

# confusion matrix
confusion = np.zeros((10, 10), dtype=int)
for tl, pl in zip(test_labs, preds):
    confusion[tl][pl] += 1

# ---------------- confusion matrix plot ----------------
if args.plot_confusion:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')
    ax.set_title(f'{model_type.upper()} Confusion Matrix (acc={clean_acc * 100:.1f}%)', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            color = 'white' if confusion[i][j] > confusion.max() / 2 else 'black'
            ax.text(j, i, str(confusion[i][j]), ha='center', va='center', color=color, fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(script_dir, args.save_dir, f'confusion_{model_type}.png')
    fig.savefig(cm_path, dpi=200, bbox_inches='tight')
    print(f"Confusion matrix saved: {cm_path}")
    plt.close(fig)

# ---------------- LaTeX table: clean results ----------------
print("\n% ---------- LaTeX: Per-class Accuracy ----------")
print(r"\begin{table}[H]")
print(r"\centering")
print(r"\caption{" + f"{model_type.upper()} Per-Class Test Accuracy" + r"}")
print(r"\begin{tabular}{@{}cc@{}}")
print(r"\toprule")
print(r"Digit & Accuracy \\ \midrule")
for cls in range(10):
    print(f"{cls} & {per_class[cls] * 100:.1f}\\% \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# ======================================================================
# Robustness Analysis
# ======================================================================
if not args.robustness:
    exit()

translation_levels = [0, 1, 2, 3, 4, 5]
rotation_levels = [0, 5, 10, 15, 20, 30]
noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.3]
results = {}


def run_perturbation(name, levels, perturb_fn):
    accs = []
    base_imgs = test_imgs_4d.copy() if model_type == 'cnn' else test_imgs.copy()
    for lvl in levels:
        if lvl == 0:
            accs.append(clean_acc)
        else:
            np.random.seed(309)
            perturbed = perturb_fn(base_imgs, lvl)
            acc = evaluate(perturbed, test_labs)
            accs.append(acc)
            print(f"  {name} lvl={lvl}: {acc * 100:.2f}%")
    results[name] = (levels, accs)


print("\n--- Robustness Analysis ---")

print("\n[1] Translation")
run_perturbation("Translation", translation_levels,
                 lambda x, s: nn.augment.random_shift(
                     x if x.ndim == 4 else x.reshape(-1, 1, 28, 28), max_shift=s))

print("\n[2] Rotation")
run_perturbation("Rotation", rotation_levels,
                 lambda x, d: nn.augment.random_rotation(
                     x if x.ndim == 4 else x.reshape(-1, 1, 28, 28), max_deg=d))

print("\n[3] Gaussian Noise")
run_perturbation("GaussianNoise", noise_levels,
                 lambda x, std: nn.augment.random_noise(
                     x if x.ndim == 4 else x.reshape(-1, 1, 28, 28), std=std))

# ---------------- LaTeX table: robustness ----------------
print("\n% ---------- LaTeX: Robustness Table ----------")
print(r"\begin{table}[H]")
print(r"\centering")
print(r"\caption{" + f"{model_type.upper()} Robustness under Perturbations" + r"}")
print(r"\begin{tabular}{@{}l" + "c" * (len(translation_levels) - 1) + r"@{}}")
print(r"\toprule")
print(r"Perturbation & " + " & ".join(f"{lvl}" for lvl in translation_levels[1:]) + r" \\ \midrule")

for name, (levels, accs) in results.items():
    vals = " & ".join(f"{a * 100:.1f}\\%" for a in accs[1:])  # skip clean (lvl 0)
    print(f"{name} & {vals} \\\\")

# accuracy drop row
print(r"\midrule")
for name, (levels, accs) in results.items():
    drops = " & ".join(f"{(clean_acc - a) * 100:+.1f}" for a in accs[1:])
    print(f"{name} drop & {drops} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# ---------------- summary ----------------
print(f"\n{'=' * 55}")
print(f"Clean accuracy: {clean_acc * 100:.2f}%")
print("Accuracy under strongest perturbation:")
for name, (levels, accs) in results.items():
    print(f"  {name} ({levels[-1]}): {accs[-1] * 100:.2f}% (drop: {(clean_acc - accs[-1]) * 100:.1f}pp)")
print("Done.")
