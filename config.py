import os
import torch
import multiprocessing

DATA_ROOT = 'rocks'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

MODEL_NAMES = ['resnet18', 'resnet50', 'squeezenet', 'shufflenetv2', 'mobilenetv3']
ATTENTION_TYPES = ['cbam', 'se', 'eca', 'spatial', 'channel', 'none']  
DEFAULT_ATTENTION = 'cbam'  
TRAIN_MULTIPLE_MODELS = True  

BATCH_SIZE = 32
NUM_EPOCHS = 100  
LEARNING_RATE = 3e-4  
WEIGHT_DECAY = 0.0003  
LABEL_SMOOTHING = 0.1  
OPTIMIZER = 'AdamW'  
LR_SCHEDULER = 'OneCycleLR'  
PCT_START = 0.3  


NUM_WORKERS = min(8, multiprocessing.cpu_count())


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  
CUDA_VISIBLE_DEVICES = '0'  
CUDNN_BENCHMARK = True  
CUDNN_DETERMINISTIC = False  
MIXED_PRECISION = True  
PIN_MEMORY = True  
NON_BLOCKING = True  
PREFETCH_FACTOR = 2  


IMG_SIZE = 384  
CROP_SIZE = 299  
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


COLOR_JITTER = {
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1
}
RANDOM_HORIZONTAL_FLIP_PROB = 0.5
RANDOM_VERTICAL_FLIP_PROB = 0.3
RANDOM_ROTATION_DEGREES = 30
RANDOM_ERASING_PROB = 0.1
RANDOM_ERASING_SCALE = (0.02, 0.1)


OUTPUT_DIR = 'output'  


SAVE_FREQ = 5  


if DEVICE == 'cuda':
    
    if CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    
    
    if CUDA_VISIBLE_DEVICES:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES 