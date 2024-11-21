# Distribution-induced Bidirectional Generative Adversarial Network for Graph Representation Learning ([DBGAN](https://arxiv.org/abs/1912.01899))
  For an extended period, I have been focused on reproducing the code from the DBGAN paper. The original code from the paper was written in **TensorFlow** and is based on a very outdated version. Building on this, I have reimplemented the code using **PyTorch**. The main focus of this paper is graph representation learning, which involves mapping data into a low-dimensional space for representation. The entire process is illustrated in the diagram below.
<img src="/images/DBGAN.png" alt="vis" width="900"/>

## My work
I used the DBGAN proposed in the original paper as an upstream task to learn the features of bearing fault data (based on graph representation learning). For the downstream task, I created a very simple network structure aimed at performing fault diagnosis using the features learned from the upstream task.

## Deficiencies
Due to my limited capabilities, the reproduced code may still contain certain issues, resulting in a final fault diagnosis accuracy of only 90%. Below is the training loss during the DBGAN training process:
<img src="/images/loss.png" alt="vis" width="900"/>
For the **DBGAN code**, please refer to [DBGAN](optim_wgan.py)
