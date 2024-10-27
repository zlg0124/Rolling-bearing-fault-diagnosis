# Generative Adversarial Nets ([GAN](https://arxiv.org/abs/1406.2661))

## Generator and Discruminator
* $G(x;\theta_g)$, where **G** is a diffrerntiable function represent by MLP with parameters $\theta_g$;

* $D(x;\theta_d)$ that outputs a **single scalar**

## Value function $V(G, D)$
* 以分类任务问题进行讨论:
若把真正的图片看做1，而生成的图片看做0,则我们可以很清楚的认识到: 若$D$训练的特别好，$D(G(z))=0$,则$log(1-D(G(z)))=log1=0$, 此时$D(x)=1$, $log(D(x))=1$, 整个的$V(G, D)=0$;
* 根据上述分析，我们首先训练D,保证$\max _D V(D, G)$；
* 其次训练G,使得$\min _G V(G, D)$, 即逐渐减小$log(1-D(z))$，原因在于使得D无法分辨生成的图片到底是0还是1，避免出现D(G(z))=1的情况。因为，在极端情况下, $D(G(z))=1$, 此时$log(1-D(G(z)))=log0~-\infty$,这就说明D没有得到完全的训练。


