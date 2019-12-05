# basic_vqa
Pytorch implementation of the paper - VQA: Visual Question Answering (https://arxiv.org/pdf/1505.00468.pdf).

![model](./png/basic_model.png)

## Usage 

#### 1. Clone the repositories.
```bash
$ git clone https://github.com/tbmoon/basic_vqa.git
```

#### 2. Download and Preproccess VQA Datasets

```bash
$ cd basic_vqa/utils
$ ./preprocess.sh
```

#### 3. Train model for VQA task.

```bash
$ cd ..
$ python3 train.py
```

## Results

- Comparison Result

| Model | Metric | Dataset | Accuracy | Source |
| --- | --- | --- | --- | --- |
| Paper Model | Open-Ended | VQA v2 | 54.08 | [VQA Challenge](https://visualqa.org/roe.html) |
| My Model | Multiple Choice | VQA v2 | **54.72** | |


- Loss and Accuracy on VQA datasets v2

![train1](./png/train.png)


## References
* Paper implementation
  + Paper: VQA: Visual Question Answering
  + URL: https://arxiv.org/pdf/1505.00468.pdf
    
* Pytorch tutorial
  + URL: https://pytorch.org/tutorials/
  + Github: https://github.com/yunjey/pytorch-tutorial
  + Github: https://github.com/GunhoChoi/PyTorch-FastCampus

* Preprocessing
  + Tensorflow implementation of N2NNM
  + Github: https://github.com/ronghanghu/n2nmn
