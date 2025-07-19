
## Environmental requirements

```
python >= 3.6
pytorch >= 1.7
torchvision >= 0.8
numpy
matplotlib
tqdm
```

## Install dependencies

```bash

pip install torch torchvision numpy matplotlib tqdm


pip install torch-directml
```

## Usage instructions

### Train a single model

```bash
python main.py --model resnet18

python main.py --model resnet50 --attention se

python main.py --model mobilenetv3 --attention eca

python main.py --model squeezenet --attention none




### Perform attention mechanism ablation experiments

```bash
# Perform attention mechanism ablation experiments on the ResNet18 model (train all attention types)
python main.py --model resnet18 --attention-ablation
