import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import pickle
import os
from DPP import distribution
from utils import sparse_mx_to_torch_sparse_tensor, _gradient_penalty
import  matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('D:\\DL_project\\bearning_fault\\DBGAN\\data.pkl', 'rb') as f:
    feas = pickle.load(f)

save_path = 'D:\\DL_project\\bearning_fault\\DBGAN'



# parameters
in_features = feas['features'].shape[1]
hidden1 = 128
hidden2 = 256
hidden3 = 64


# model
discriminator = Discriminator(in_dim=256, hidden1=hidden1, hidden3=hidden3).to(device)
D_Graph = D_graph(in_features, hidden2).to(device)
model = Encoder(in_features, hidden1, hidden2).to(device)
model_z2g = Generator_z2g(in_features, hidden1, hidden2).to(device)

# Optimizer
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.99))
discriminator_optimizer_z2g = torch.optim.Adam(D_Graph.parameters(), lr=0.0001, betas=(0.9, 0.99))

generator_optimizer_z2g = torch.optim.Adam(model_z2g.parameters(), lr=0.0001, betas=(0.9, 0.99))
encoder_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99)) 

optimizers = {
    'generator_optimizer_z2g': generator_optimizer_z2g,
    'encoder_optimizer': encoder_optimizer,
    'discriminator_optimizer': discriminator_optimizer,
    'discriminator_optimizer_z2g': discriminator_optimizer_z2g
}

# DPP
kde = distribution(feas, n_components=256)

def train(model, model_z2g, discriminator, D_Graph, optimizers, distribution, data, device):

    # unzip optimizer
    discriminator_optimizer = optimizers['discriminator_optimizer']
    discriminator_optimizer_z2g = optimizers['discriminator_optimizer_z2g']
    generator_optimizer_z2g = optimizers['generator_optimizer_z2g']
    encoder_optimizer = optimizers['encoder_optimizer']

    # unzip data 
    features_dense = torch.FloatTensor(data['features']).to(device)
    pos_weight = torch.tensor(data['pos_weight'], dtype=torch.float32).to(device)
    norm = torch.tensor(data['norm'], dtype=torch.float32).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(data['adj_norm']).to(device)
    # adj_orig = sparse_mx_to_torch_sparse_tensor(data['adj_orig']).to(device)
    adj_orig = sparse_mx_to_torch_sparse_tensor(data['adj_orig']).to(device)

    

    # DPP 
    z_real_dist = distribution.sample(adj.shape[0]) # 2708, 128
    z_real_dist = torch.FloatTensor(z_real_dist).to(device)
    

    for _ in range(5):
    
        # D_graph == Dx
        discriminator_optimizer_z2g.zero_grad()
        real_data = features_dense
        GD_real = D_Graph(features_dense)
        z2g = model_z2g(z_real_dist, adj)
        GD_fake = D_Graph(z2g.detach())
        generated_data = z2g
        gradient_penalty = _gradient_penalty(real_data, generated_data, D_Graph, device=device)
        GD_loss = GD_fake.mean() - GD_real.mean() + gradient_penalty
        GD_loss.backward()
        discriminator_optimizer_z2g.step()

    # Generator
    generator_optimizer_z2g.zero_grad()
    z2g = model_z2g(z_real_dist, adj)
    GD_fake = D_Graph(z2g) # 2708, 1
   
    generator_loss_z2g = -GD_fake.mean()
    generator_loss_z2g.backward()
    generator_optimizer_z2g.step()

   
    # Discriminator == Dz
    discriminator_optimizer.zero_grad()
    real_data = z_real_dist
    generated_data, _ = model(features_dense, adj)
    gradient_penalty = _gradient_penalty(real_data, generated_data, discriminator, device=device)

    d_fake = discriminator(generated_data.detach())
    d_real = discriminator(real_data)
    discriminator_loss = d_fake.mean() - d_real.mean() + gradient_penalty
    discriminator_loss.backward()
    discriminator_optimizer.step()


    # Encoder
    encoder_optimizer.zero_grad()
    embeddings, preds_sub = model(features_dense, adj)
    labels_sub = adj_orig.to_dense().reshape(-1)

    cost = norm * F.binary_cross_entropy_with_logits(preds_sub, labels_sub, pos_weight)

    preds_cycle = model_z2g(embeddings, adj)
    labels_cycle = features_dense
    
    cost_cycle = norm * F.binary_cross_entropy_with_logits(preds_cycle, labels_cycle)
    reconstruction_loss = 0.01*cost + cost_cycle # Reconstruction loss
    
    latent_dim, preds_sub = model(features_dense, adj)
    d_fake = discriminator(latent_dim)
    encoder_A_loss = -d_fake.mean()

    encoder_total_loss = encoder_A_loss + 0.01 * reconstruction_loss
    
    encoder_total_loss.backward()
    encoder_optimizer.step()


    all_loss = [encoder_total_loss.item(), generator_loss_z2g.item(),  GD_loss.item(), discriminator_loss.item()]
    emb = model.embeddings

    return all_loss, emb

if __name__ == '__main__':
    num_epochs = 100
    emb_lst = []
    encoder_loss_lst = []
    generator_loss_lst = []
    GD_loss_lst = []
    disc_loss_lst = []
    for epoch in range(num_epochs):
        all_loss, emb = train(model, model_z2g, discriminator, D_Graph, optimizers, kde, feas, device)
        print('encoder_total_loss: {:.4f}, generator_loss_z2g: {:.4f},  GD_loss: {:.4f}, discriminator_loss: {:.4f}'.format(all_loss[0], all_loss[1], all_loss[2], all_loss[3]))

        encoder_loss_lst.append(all_loss[0])
        generator_loss_lst.append(all_loss[1])
        GD_loss_lst.append(all_loss[2])
        disc_loss_lst.append(all_loss[3])

        emb_lst.append(emb.detach().cpu())
    

    data_dict = {'emb': emb_lst}
    emb_data = os.path.join(save_path, 'emb_lst.pkl')

    with open(emb_data, 'wb') as f:
        pickle.dump(data_dict, f)
    f.close() 

    
    epochs = np.arange(100)
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, np.array(encoder_loss_lst), label='Encoder Loss')
    plt.plot(epochs, np.array(generator_loss_lst), label='Generator Loss')
    plt.plot(epochs, np.array(GD_loss_lst), label="GD Loss")
    plt.plot(epochs, np.array(disc_loss_lst), label="Discriminator Loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss of Traning')
    plt.legend()
    plt.show()





        


    





 




    

        
    

