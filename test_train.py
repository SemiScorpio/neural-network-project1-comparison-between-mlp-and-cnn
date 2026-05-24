import argparse
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'],
                    help='Model type: mlp or cnn')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--log_iters', type=int, default=100, help='Logging iterations')
parser.add_argument('--aug_shift', type=int, default=0, metavar='N',
                    help='Random shift augmentation (max pixels, 0=disabled)')
parser.add_argument('--aug_noise', type=float, default=0.0, metavar='STD',
                    help='Gaussian noise augmentation (std, 0=disabled)')
parser.add_argument('--aug_rotate', type=float, default=0.0, metavar='DEG',
                    help='Random rotation (max degrees, 0=disabled)')
parser.add_argument('--aug_resize', type=float, default=0.0, metavar='FRAC',
                    help='Random resize range, e.g. 0.1 means [0.9, 1.1] (0=disabled)')
parser.add_argument('--aug_hflip', action='store_true', help='Random horizontal flip')
parser.add_argument('--dropout', type=float, default=0.0, metavar='P',
                    help='Dropout probability (0 = disabled)')
args = parser.parse_args()

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0

# ---------------------- model selection ----------------------
print(f"Using model: {args.model}, epochs: {args.epochs}, lr: {args.lr}, batch_size: {args.batch_size}")

if args.model == 'mlp':
    model = nn.models.Model_MLP([train_imgs.shape[-1], 256, 10], 'ReLU',
                                dropout=args.dropout)

elif args.model == 'cnn':
    train_imgs = train_imgs.reshape(-1, 1, 28, 28)
    valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)
    model = nn.models.Model_CNN(dropout=args.dropout)

optimizer = nn.optimizer.SGD(init_lr=args.lr, model=model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1000, 2500, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max() + 1)

# compose augmentation function from flags
augment_fn = None
aug_parts = []
if args.aug_shift > 0:
    aug_parts.append(f'shift={args.aug_shift}')
if args.aug_noise > 0:
    aug_parts.append(f'noise={args.aug_noise}')
if args.aug_hflip:
    aug_parts.append('hflip')
if args.aug_rotate > 0:
    aug_parts.append(f'rotate={args.aug_rotate}')
if args.aug_resize > 0:
    aug_parts.append(f'resize={args.aug_resize}')
if aug_parts:
    shift, noise, hflip = args.aug_shift, args.aug_noise, args.aug_hflip
    rotate, resize = args.aug_rotate, args.aug_resize
    augment_fn = lambda x: nn.augment.compose_augment(
        x, shift=shift, noise=noise, hflip=hflip, rotate=rotate, resize=resize)
    print(f"Augmentation: {', '.join(aug_parts)}")

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                           batch_size=args.batch_size, scheduler=scheduler,
                           augment_fn=augment_fn)

save_dir = f'./best_models_{args.model}'
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
             num_epochs=args.epochs, log_iters=args.log_iters, save_dir=save_dir)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()
