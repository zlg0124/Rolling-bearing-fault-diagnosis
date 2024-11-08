import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import pickle
from DPP import distribution
from utils import sparse_mx_to_torch_sparse_tensor
torch.autograd.set_detect_anomaly(True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('/Users/wangjun/bearning_fault/Demo_GCN/data/cora/data.pkl', 'rb') as f:
    feas = pickle.load(f)
f.close()


# parameters
in_features = feas['features_dense'].shape[1]
hidden1 = 64
hidden2 = 128
hidden3 = 64
lr = 0.001

# model
discriminator = Discriminator(in_dim=128, hidden1=hidden1, hidden3=hidden3).to(device)
D_Graph = D_graph(in_features, hidden2).to(device)
model = Encoder(in_features, hidden1, hidden2).to(device)
model_z2g = Generator_z2g(in_features, hidden1, hidden2).to(device)

# Optimizer
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.99))
discriminator_optimizer_z2g = torch.optim.Adam(D_Graph.parameters(), lr=0.001, betas=(0.9, 0.99))

generator_optimizer_z2g = torch.optim.Adam(model_z2g.parameters(), lr=0.001, betas=(0.9, 0.99))
generator_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
reconstrcution_optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, betas=(0.9, 0.99)) # recontruction optim
optimizers = {
    'generator_optimizer_z2g': generator_optimizer_z2g,
    'generator_optimizer': generator_optimizer,
    'reconstruction_optimizer': reconstrcution_optimizer,
    'discriminator_optimizer': discriminator_optimizer,
    'discriminator_optimizer_z2g': discriminator_optimizer_z2g
}

# DPP
kde = distribution(feas)

def train(model, model_z2g, discriminator, D_Graph, optimizers, distribution, data, device):

    # unzip optimizer
    discriminator_optimizer = optimizers['discriminator_optimizer']
    discriminator_optimizer_z2g = optimizers['discriminator_optimizer_z2g']
    generator_optimizer_z2g = optimizers['generator_optimizer_z2g']
    generator_optimizer = optimizers['generator_optimizer']
    reconstruction_optimizer = optimizers['reconstruction_optimizer']

    # unzip data 
    # transform tensor
    adj = sparse_mx_to_torch_sparse_tensor(data['adj']).to(device)
    # adj_norm = data['adj_norm']
    # adj_label = data['adj_label']
    # adj_dense = torch.FloatTensor(data['adj_dense']).to(device)
    features_dense = torch.FloatTensor(data['features_dense']).to(device)
    pos_weight = torch.tensor(data['pos_weight'], dtype=torch.float32).to(device)
    norm = torch.tensor(data['norm'], dtype=torch.float32).to(device)
    adj_orig = sparse_mx_to_torch_sparse_tensor(data['adj_orig']).to(device)
    
    # transform
    features = data['features']
    indices = torch.from_numpy(features[0].T).to(torch.long)
    values = torch.from_numpy(features[1])
    shape = torch.Size(features[2])
    features = torch.sparse_coo_tensor(indices, values, shape).to(device)

    # DPP 
    z_real_dist = distribution.sample(adj.shape[0]) # 2708, 128
    # z_real_dist = np.random.randn(2708, 128) # test
    z_real_dist = torch.FloatTensor(z_real_dist).to(device)
    # 2708, 1433


    for _ in range(5):

        reconstrcution_optimizer.zero_grad()

        embeddings, preds_sub = model(features, adj)
        labels_sub = adj_orig.to_dense().reshape(-1)

        cost = norm * F.binary_cross_entropy_with_logits(preds_sub, labels_sub, pos_weight)
        preds_cycle = model_z2g(embeddings, adj)
        labels_cycle = features_dense
        cost_cycle = norm * torch.mean(torch.square(preds_cycle - labels_cycle))
        reconstruction_loss = 0.01 * cost + cost_cycle # Reconstruction loss
        
        reconstruction_loss.backward()
        reconstruction_optimizer.step()

        # D_graph
        z2g = model_z2g(z_real_dist, adj)
        GD_fake = D_Graph(z2g) # 2708, 1
        # GD_real = D_Graph(features_dense)
        generator_optimizer_z2g.zero_grad()
        generator_loss_z2g = F.binary_cross_entropy_with_logits(GD_fake, torch.ones_like(GD_fake)) # 对应原始代码中GG_loss
        generator_loss_z2g.backward()
        generator_optimizer_z2g.step()
    
    # Generator 
    latent_dim, _ = model(features, adj)
    generator_optimizer.zero_grad()
    d_fake = discriminator(latent_dim)
    d_real_label = torch.ones_like(d_fake).to(device)
    generator_loss = F.binary_cross_entropy_with_logits(d_fake, d_real_label)
    generator_loss.backward()
    generator_optimizer.step()


    # D_graph判别器训练
    GD_real = D_Graph(features_dense)
    GD_fake = D_Graph(model_z2g(z_real_dist, adj))
    discriminator_optimizer_z2g.zero_grad()
    GD_loss_real = F.binary_cross_entropy_with_logits(GD_real, torch.ones_like(GD_real))
    GD_loss_fake = F.binary_cross_entropy_with_logits(GD_fake.detach(), torch.zeros_like(GD_fake))
    GD_loss = (GD_loss_real + GD_loss_fake) / 2.0
    GD_loss.backward()
    discriminator_optimizer_z2g.step()
    
    # Discriminator
    discriminator_optimizer.zero_grad()
    d_real = discriminator(z_real_dist)
    d_real_label = torch.ones_like(d_real).to(device)
    d_fake_label = torch.zeros_like(d_fake).to(device)
    discriminator_loss_real = F.binary_cross_entropy_with_logits(d_real, d_real_label)
    discriminator_loss_fake = F.binary_cross_entropy_with_logits(d_fake.detach(), d_fake_label)
    discriminator_loss = (discriminator_loss_fake + discriminator_loss_real) / 2.0
    discriminator_loss.backward()
    discriminator_optimizer.step()
    
    total_loss = [reconstruction_loss.item(), generator_loss_z2g.item(), generator_loss.item(), GD_loss.item(), discriminator_loss.item()]

    return total_loss

if __name__ == '__main__':
    for epoch in range(10):
        total_loss = train(model, model_z2g, discriminator, D_Graph, optimizers, kde, feas, device)
        print('reconstruction_loss: {:.4f}, generator_loss_z2g: {:.4f}, generator_loss: {:.4f}, GD_loss: {:.4f}, discriminator_loss: {:.4f}'.format(total_loss[0], 
                                                                                                                                                    total_loss[1],
                                                                                                                                                    total_loss[2],
                                                                                                                                                    total_loss[3],
                                                                                                                                                    total_loss[4]))
    

