from random import random

import numpy as np
import torch



DATA_MODES = ['train', 'val', 'test']
RESCALE_SIZE = 224
SEED=20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
TRAIN_DIR = Path('../input/journey-springfield/train/simpsons_dataset/')
TEST_DIR = Path('../input/journey-springfield/testset/testset/')

train_val_files = sorted(list(TRAIN_DIR.rglob('*/*.jpg')))
test_files = sorted(list(TEST_DIR.rglob('*.jpg')))