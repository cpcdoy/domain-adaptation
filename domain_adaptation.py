import os
import sys
import argparse
import math
import time
import tqdm

import torch
from torch import nn

import numpy as np

import sklearn.metrics as skm

import matplotlib.pyplot as plt

import models, datasets


class DA:
    """
    Performs Domain Association from SVHN to MNIST
    """

    def __init__(self):
        """
        Initialize the DA class
        """
        
        self.parser = self.init_cmd_parser()
        self.args = self.parser.parse_args()
        
        self.gpu = 0
        
        # List of datasets with their batch size
        self.dataset_list = [('mnist', self.args.batch_s), ('svhn', self.args.batch_t)]
        
    def set_gpu(self, gpu):
        """
        Sets the current GPU
        
        Parameters:
		-gpu (int): The index of the GPU to use
        
        Returns:
        -Nothing
        """
        
        self.gpu = gpu
        
    def to_gpu(self, x):
        """
        Sets an item as usable by the GPU
        
        Parameters:
		-x (torch obj): Any torch to upload to the GPU
	
		Returns:
		-The object usable on GPU
        """
        
        return x.cuda(self.gpu)
    
    def to_cpu(self, x):
        """
        Sends an item to the CPU from the GPU
        
        Parameters:
		-x (torch obj): Any torch to download to the CPU
	
		Returns:
		-The object usable on CPU
        """
        
        return x.cpu().detach().numpy()

    def init_cmd_parser(self):
        """
        Init the cmd line parser
        
        Parameters:
		-Nothing
        
		Returns:
		-The parser
        """
    
        parser = argparse.ArgumentParser(description='SVHN to MNIST domain adaptation')

        parser.add_argument('--resume', default="", help="Weights path to resume from")
        parser.add_argument('--data', default="data/", help="Datasets path")
        parser.add_argument('--save', default="save/", help="Result path")
        parser.add_argument('--eval', default=0, type=int, help="Evaluate model")
        # These settings take ~14.5GB of memory
        parser.add_argument('--batch_s', default=100, type=int, help="Source domain batch size")
        parser.add_argument('--batch_t', default=1000, type=int, help="Target domain batch size")
        parser.add_argument('--disp', action='store_true', help="Display predictions during eval")
    
        return parser
        
    def process(self):
        """
        Start the whole process of either training or evaluation the model
        
        Parameters:
		-Nothing
        
		Returns:
		-The parser
        """
    
        train = self.args.eval == 0
        # Setup the model and either load weights or start fresh
        self.model = models.self_ensembling_model()
        if os.path.exists(self.args.resume):
            print("Using weights from \"{}\"".format(self.args.resume))
            self.model.load_state_dict(torch.load(self.args.resume))
        
        self.model = self.to_gpu(self.model)
            
        # Directory to save the trained weights in
        if train:
            os.makedirs(self.args.save, exist_ok=True)
        
        # Load datasets
        self.get_datasets(train)
        if train:
            self.train()
        else:
            self.eval()
        
    def get_datasets(self, train=True):
        """
        Gets the datasets from the "datasets" module
        
        Parameters:
		-train (bool, default=True): either load train or eval data
        
		Returns:
		-Nothing
        """
        
        data = datasets.load_datasets(self.args.data, train)
        self.datasets = {}
        # Convert to DataLoader
        for (i, batch_size) in self.dataset_list:
            self.datasets[i] = torch.utils.data.DataLoader(data[i], batch_size=batch_size, shuffle=True)

    def train(self, nb_epochs=100):
        """
        Train the model
        
        Parameters:
		-nb_epochs (bool, default=True): Max number of epochs
        
		Returns:
		-Nothing
        """
        
        # Gather DataLoaders
        train = self.datasets['svhn']
        val = self.datasets['mnist']
    
        # Set Adam as optimizer with same params as in the paper
        optim = torch.optim.Adam(self.model.parameters(), lr=3e-4, betas=(0.5, 0.999), amsgrad=True)
    
        # Init losses
        assoc_loss = models.assoc_loss()
        classification_loss = nn.CrossEntropyLoss()
        
        # Set model to train
        self.model.train()

        # Timer
        start_time = time.time()
    
        # Log losses, accuracy
        num_iter = 0
        train_hist = {}
        train_hist['classification_loss'] = []
        train_hist['assoc_loss'] = []
        train_hist['acc_s'] = []
        train_hist['acc_t'] = []
        
        clock = time.time()
        
        for epoch in range(nb_epochs):
            epoch_start_time = time.time()
    
            # Get new batches every iteration
            pbar_batch = tqdm.tqdm()
            for (xs, ys), (xt, yt) in zip(*(self.datasets['svhn'], self.datasets['mnist'])):
    
                # Batches to GPU
                xs = self.to_gpu(xs)
                ys = self.to_gpu(ys)
                xt = self.to_gpu(xt)
                yt = self.to_gpu(yt)
    
                self.model.zero_grad()
    
                # Get embeddings and classification predictions
                phi_s, yp   = self.model(xs)
                phi_t, ypt  = self.model(xt)
    
                # Remove unnecessary dims
                yp  = yp.squeeze().clone()
                ypt = ypt.squeeze().clone()
    
                # Log losses and compute them
                train_hist['classification_loss'].append(classification_loss(yp, ys).mean())
                train_hist['assoc_loss'].append(assoc_loss(phi_s, phi_t, ys).mean())
                
                # Compute accuracies
                ypt_max = ypt.max(dim=1)[1]
                acc_s = torch.eq(yp.max(dim=1)[1], ys).sum().float()  / train.batch_size
                acc_t = torch.eq(ypt_max, yt).sum().float() / val.batch_size
                
                # Log accuracies
                train_hist['acc_s'].append(self.to_cpu(acc_s))
                train_hist['acc_t'].append(self.to_cpu(acc_t))
                
                # Comined loss
                combined_loss = train_hist['classification_loss'][-1] + train_hist['assoc_loss'][-1]
                # Compute gradients
                combined_loss.backward()
                
                #scheduler.step()
                # Step optimizer by learning rate
                optim.step()
    
                num_iter += 1
    
                if num_iter % 10 == 0:
                
                    # Compute accuracies from last 100
                    acc_s = np.mean(train_hist['acc_s'][-100:])
                    acc_t = np.mean(train_hist['acc_t'][-100:])
    
                    # Predictions and gt to cpu
                    yt_cpu = self.to_cpu(yt)
                    ypt_cpu = self.to_cpu(ypt_max)
                    
                    # Display stats
                    pbar_batch.set_description('Epoch {}, Iteration {} - S {:.3f} % - T {:.3f} % - F1: {:.3f} - Precision: {:.3f} - Recall: {:.3f}'.format(epoch, num_iter,acc_s*100,acc_t*100, skm.f1_score(yt_cpu, ypt_cpu, average="weighted"),skm.precision_score(yt_cpu, ypt_cpu, average="weighted"), skm.recall_score(yt_cpu, ypt_cpu, average="weighted")))
    
                if time.time() - clock > 60:
    
                    clock = time.time()
    
                    # Save model
                    torch.save(self.model.state_dict(), os.path.join(self.args.save, 'se_weights_epoch_{}.pth'.format(epoch)))
    
    
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            
    def eval(self):
        """
        Evaluate the model by using eval datasets,
        displaying images and the predicted labels for both 
        the source and target domain
        
        Parameters:
		-Nothing
        
		Returns:
		-Nothing
        """
    
        # Get eval datasets
        nb_preds = self.args.eval
        train = self.datasets['svhn']
        val = self.datasets['mnist']
    
        # Set model to evaluation mode
        self.model.eval()
        
        pbar_batch = tqdm.tqdm()
        acc_s = 0
        acc_t = 0
        i = 0
        # Get a new batch each iteration
        for (xs, ys), (xt, yt) in zip(*(self.datasets['svhn'], self.datasets['mnist'])):
        
            # Batches to GPU
            i += 1
            xs = self.to_gpu(xs)
            ys = self.to_gpu(ys)
            xt = self.to_gpu(xt)
            yt = self.to_gpu(yt)
        
            # Predict labels
            _, yp   = self.model(xs)
            _, ypt  = self.model(xt)
            
            # Compute accuracy
            ypt_max = ypt.max(dim=1)[1]
            yp_max = yp.max(dim=1)[1]
            acc_s_curr = torch.eq(yp_max, ys).sum().float() / train.batch_size
            acc_t_curr = torch.eq(ypt_max, yt).sum().float() / val.batch_size
            
            # Accumulate accuracy
            acc_s += acc_s_curr
            acc_t += acc_t_curr
            
            # Print predictions and gt
            print("")
            print('\nPrediction source:', yp_max)
            print('Expected source:', ys)
            print('Prediction target:', ypt_max)
            print('Expected target:', yt)
            
            # Download to CPU to display stats
            yt_cpu = self.to_cpu(yt)
            ypt_cpu = self.to_cpu(ypt_max)
            
            # Print stats
            print('Stats: S {:.3f} % - T {:.3f} % - F1: {:.3f} - Precision: {:.3f} - Recall: {:.3f}'.format(acc_s_curr*100,acc_t_curr*100, skm.f1_score(yt_cpu, ypt_cpu, average="weighted"),skm.precision_score(yt_cpu, ypt_cpu, average="weighted"), skm.recall_score(yt_cpu, ypt_cpu, average="weighted")))
            
            # Display global average stats
            if i == nb_preds or self.args.disp:
                pbar_batch.set_description('Average stats: S {:.3f} % - T {:.3f} % - F1: {:.3f} - Precision: {:.3f} - Recall: {:.3f}'.format(acc_s*100/nb_preds,acc_t*100/nb_preds, skm.f1_score(yt_cpu, ypt_cpu, average="weighted"),skm.precision_score(yt_cpu, ypt_cpu, average="weighted"), skm.recall_score(yt_cpu, ypt_cpu, average="weighted")))
                if not self.args.disp:
                    break
                else:
                    nb_preds += 1
                
            # Display samples if "--disp" is set
            if self.args.disp:
                fig = plt.figure( figsize=(40, 40) )
                for i in range(1, self.args.batch_s):
                    ax = fig.add_subplot(self.args.batch_s, self.args.batch_s + 1, i, title='Pred(' + str(yp_max[i - 1].item()) + ')\n Label(' + str(ys[i - 1].item()) + ')')
                    ax.axis('off')
                    # Swap axes because matlplotlib wants HxWxC then clamp to avoid overflows
                    ax.imshow(self.to_cpu(xs[i - 1].permute(1, 2, 0).clamp(0.0, 1.0)))
                    
                    ax = fig.add_subplot(self.args.batch_t, self.args.batch_t + 1, self.args.batch_s + 1 + i, title='Pred(' + str(ypt_max[i - 1].item()) + ')\n Label(' + str(yt[i - 1].item()) + ')')
                    ax.axis('off')
                    # Swap axes because matlplotlib wants HxWxC then clamp to avoid overflows
                    ax.imshow(self.to_cpu(xt[i - 1].permute(1, 2, 0).clamp(0.0, 1.0)))
                
                plt.subplots_adjust(hspace=1.0)
                plt.show()
            
            # Clean PyTorch cache to avoid CUDA memory error
            del xs
            del xt
            del ys
            del yt
            torch.cuda.empty_cache()
            
                

if __name__ == '__main__':

    domain_assoc = DA()            
    domain_assoc.process()
