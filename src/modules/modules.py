#Импорт модулей
import os, sys
import time
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
from pathlib import Path
from multiprocessing.pool import ThreadPool
import PIL
from PIL import Image
from skimage import io
from sklearn.preprocessing import LabelEncoder
import torch
import copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from tqdm import tqdm, tqdm_notebook
import pickle
from sklearn.model_selection import train_test_split
