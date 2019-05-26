import torch
from torch import nn

import torch.nn.functional as F

class walker_loss(nn.Module):
    """
    The Walker Loss implemented using eq. 2, 3, 4, 5 from [P. Haeusser, 2017]
    
    Changes: using KL div instead of cross-entropy because CE only
    accepts raw, unormalized scores for each class.
    It's basically the same with one less regularization term (entropy).
    """

    def forward(self, P_s__t__s, y):
        """
        Compute the walker loss
        
        Two-step round_trip probability of a random walker
        from source to target baack to source.
        
        The probabilities are forced to be uniform.
        
        Parameters:
        -P_s__t__s: probability of the two-step round-trip
        -y: source domain labels
        
        Returns:
        -walker_loss: the final loss's value
        """
    
        equality_matrix = torch.eq(y.clone().view(-1,1), y).float()
        # Tij
        T = equality_matrix / equality_matrix.sum(dim=1, keepdim=True)
        T.requires_grad = False

        # Log probability as input
        # Prefer summing instead of averaging because of different batch sizes in source and target
        walker_loss = F.kl_div(torch.log(1e-8 + P_s__t__s), T, size_average=False)
        walker_loss /= T.size()[0]

        return walker_loss

class visit_loss(nn.Module):
    """
    The Visit Loss implemented using eq. 6, 7 from [P. Haeusser, 2017]
    
    Changes: using KL div instead of cross-entropy because CE only
    accepts raw, unormalized scores for each class.
    It's basically the same with one less regularization term (entropy).
    """

    def forward(self, P_visit):
        """
        Compute the visit loss
        
        Acts as a regularizer loss for the walker loss, forcing
        the walker to visit every target sample equaly to enable
        better generalization
        
        Parameters:
        -P_visit: probability of the two-step round-trip
        
        Returns:
        -visit_loss: the final loss's value
        """
        
        # V = 1 / |B|
        V = torch.ones([1, P_visit.size()[1]]) / float(P_visit.size()[1])
        V.requires_grad = False
        V = V.cuda()
        
        # Log probability as input
        # Prefer summing instead of averaging because of different batch sizes in source and target
        visit_loss = F.kl_div(torch.log(1e-8 + P_visit), V, size_average=False)
        visit_loss /= V.size()[0]

        return visit_loss

class assoc_loss(nn.Module):
    """
    The final association loss which combines the walker and visit losses
    implemented using eq. 8 from [P. Haeusser, 2017]\
    """
    
    def __init__(self, walker_weight = 1.0, visit_weight = 0.1):
        """
        Init the association loss
        
        The class distribution is different between source and target,
        so we give the visit loss a lower weight
        
        Parameters:
        -walker_weight (float, default=1.0): The walker weight
        -visit_weight (float, default=0.1): The visit weight
        """
        super(assoc_loss, self).__init__()

        self.walker_loss = walker_loss()
        self.visit_loss  = visit_loss()

        self.walker_weight = walker_weight
        self.visit_weight  = visit_weight

    def forward(self, A, B, y):
        """
        Compute the association loss
        
        Compute the similarity of embeddings via dot product
        Then get the transition probability from embedding A to BaseException
        Finally get all the probabilities from any Ai back to Aj.
        
        Parameters:
        -A (tensor): source embeddings
        -B (tensor): target embeddings
        -y (tensor): source gt labels
        
        Returns:
        -association_loss: the final loss's value
        """
        
        # Mij = <Ai,Bj>
        M = torch.mm(A, B.transpose(1,0))
        
        # p(Bj| Ai)
        P_s__t = F.softmax(M, dim=1) # Ns x Nt
        # p(Aj | Bi)
        P_t__s = F.softmax(M.transpose(1,0), dim=1) # Nt x Ns

        # p(Aj | Ai)
        P_s__t__s = P_s__t @ P_t__s # Ns x Ns

        # p(Bi) = P_visit = sum p(Bj| Ai)
        P_visit = torch.mean(P_s__t, dim=0, keepdim=True) # Nt

        # Return L_assoc
        return self.visit_weight * self.visit_loss(P_visit) + self.walker_weight * self.walker_loss(P_s__t__s, y)

class self_ensembling_model(nn.Module):

    """
    Model used in "Self-Ensembling for Visual Domain AdaP_tation"
    French et al.
    https://arxiv.org/pdf/1706.05208.pdf
    """

    def __init__(self):
        """
        Neural net architecture
        
        Using Group Normalization instead of batch norm from original model
        """

        super(self_ensembling_model, self).__init__()

        def conv2d(input, filters, kernel_size, pad):
            return nn.Sequential(
                nn.Conv2d(input, filters, kernel_size=kernel_size,padding=pad),
                #nn.BatchNorm2d(filters),
                nn.GroupNorm(filters // 2, filters),
                nn.ReLU()
            )

        def block(input, filters):
            return nn.Sequential(
                conv2d(input, filters, 3, 1),
                conv2d(filters, filters, 3, 1),
                conv2d(filters, filters, 3, 1)
            )

        self.features = nn.Sequential(
            block(3,128),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            block(128,256),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Dropout2d(p=0.5),
            conv2d(256, 512, 3, 0),
            conv2d(512, 256, 1, 0),
            conv2d(256, 128, 1, 0),
            nn.AvgPool2d(6, 6, padding=0)
        )

        self.classifier = nn.Sequential(nn.Linear(128, 10))

    def forward(self, x):
        """
        Apply neural net
        
        Parameters:
        -x (tensor): input batch
        
        Returns:
        -phi (tensor): input embedding/features
        -y (tensor): 
        """

        # Embeddings/features
        phi = self.features(x)
        phi = phi.view(-1,128)
        # Classification result
        y = self.classifier(phi)
        # No softmax as opposed to the original model because CE loss already does it
        #y = F.softmax(y, dim=1)
        
        return phi, y
