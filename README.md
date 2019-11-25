# Adversarial-Attacks-Pytorch

This is a lightweight repository of adversarial attacks for Pytorch.

There are popular attack methods and some utils.

## Usage

### Dependencies

- torch 1.2.0
- python 3.6

### Installation

- `pip install torchattacks` or
- `git clone https://github.com/Harry24k/adversairal-attacks-pytorch`

```python
import torchattacks
pgd_attack = torchattacks.PGD(model, eps = 4/255, alpha = 8/255)
adversarial_images = pgd_attack(images, labels)
```

### Precautions

* **WARNING** :: All images should be scaled to [0, 1] with transform[to.Tensor()] before used in attacks.
* **WARNING** :: All models should return ONLY ONE vector of `(N, C)` where `C = number of classes`.

### Attacks and Papers

The papers and the methods with a brief summary and example.
All attacks in this repository are provided as *CLASS*.
If you want to get attacks built in *Function*, please refer below repositories.

* **Explaining and harnessing adversarial examples** : [Paper](https://arxiv.org/abs/1412.6572), [Repo](https://github.com/Harry24k/FGSM-pytorch)
  - FGSM

* **Adversarial Examples in the Physical World** : [Paper](https://arxiv.org/abs/1607.02533), [Repo](https://github.com/Harry24k/AEPW-pytorch)
  - IFGSM
  - IterLL

* **Ensemble Adversarial Traning : Attacks and Defences** : [Paper](https://arxiv.org/abs/1705.07204), [Repo](https://github.com/Harry24k/RFGSM-pytorch)
  - RFGSM

* **Towards Evaluating the Robustness of Neural Networks** : [Paper](https://arxiv.org/abs/1608.04644), [Repo](https://github.com/Harry24k/CW-pytorch)
  - CW(L2)

* **Towards Deep Learning Models Resistant to Adversarial Attacks** : [Paper](https://arxiv.org/abs/1706.06083), [Repo](https://github.com/Harry24k/PGD-pytorch)
  - PGD

* **Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"** : [Paper](https://arxiv.org/abs/1907.00895)
  - APGD

### Demos

* **White Box Attack with Imagenet** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20with%20Imagenet.ipynb)): 
To make adversarial examples with the Imagenet dataset to fool [Inception v3](https://arxiv.org/abs/1512.00567). However, the Imagenet dataset is too large so in this demo, so it uses only '[Giant Panda](http://www.image-net.org/)'.

* **Black Box Attack with CIFAR10** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20with%20MNIST.ipynb)): 
This demo provides an example of black box attack with two different models. First, make adversarial datasets from a holdout model with CIFAR10 and save it as a torch dataset. Second, use the adversarial datasets to attack a target model.

* **Adversairal Training with MNIST** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20with%20MNIST.ipynb)): 
This demo shows how to do adversarial training with this repository. The MNIST dataset and a custom model are used in this code. The adversarial training is performed with PGD and FGSM is applied to test the model.


## Update Records

### ~ Version 0.3
* **New Attacks** : FGSM, IFGSM, IterLL, RFGSM, CW(LW), PGD are added.
* **Demos** are uploaded.

### Version 0.4
* **DO NOT USE** : 'init.py' is omitted.

### Version 0.5
* **Package name changed** : 'attacks' is changed to 'torchattacks'.
* **New Attacks** : APGD is added.
* **attack.py** : 'update_model' method is added.

### Version 0.6
* **Error Solved** : 
    * Before this version, even after getting an adversarial image, the model remains evaluation mode.
    * To solve this, below methods are modified.
        * '_switch_model' method is added into **attack.py**. It will automatically change model mode to the previous mode after getting adversarial images. When getting adversarial images, model is switched to evaluation mode.
        * '__call__' methods in all attack changed to forward. Instead of this, '__call__' method is added into 'attack.py'
* **attack.py** : To provide ease of changing images to uint8 from float, 'set_mode' and '_to_uint' is added.
    * 'set_mode' determines returning all outputs as 'int' OR 'flaot' through '_to_uint'.
    * '_to_uint' changes all outputs into uint8.

### Version 0.7
* **All attacks are modified**
    * clone().detach() is used instead of .data
    * torch.autograd.grad is used instead of .backward() and .grad :
        * It showed 2% reduction of computation time.
    
### Version 0.8
* **New Attacks** : RPGD is added.
* **attack.py** : 'update_model' method is depreciated. Because torch models are passed by call-by-reference, we don't need to update models.
    * **cw.py** : In the process of cw attack, now masked_select uses a mask with dtype torch.bool instead of a mask with dtype torch.uint8.