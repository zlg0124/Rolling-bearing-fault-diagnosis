# Generative Adversarial Nets ([GAN](https://arxiv.org/abs/1406.2661))

## Generator and Discruminator
* $G(x;\theta_g)$, where **G** is a diffrerntiable function represent by MLP with parameters $\theta_g$;

* $D(x;\theta_d)$ that outputs a **single scalar**

## Value function $V(G, D)$
$$
V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

