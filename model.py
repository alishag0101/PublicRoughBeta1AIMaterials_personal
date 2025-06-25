import math
import torch
import torch.nn as nn
class MultiHeadSelfAttention(nn.Module): #when defining new classes, we can have them inherit the attributes/methods of other classes by class A(B) now A is a subclass of B. In this case we want the class MultiHeadSelfAttention to have nn.Module's attributes and methods defined. nn.Module stands for neural netowrk modules and this class contains several methods and attributes that will prove to be useful.
    def __init__(self, tokenvector_dimension: int, numberofheads: int = 8, dropping_probability: float = 0.1): #tokenvectordimension refers to the dimensionality of the token vectors, numberofheads refers to the number of subspaces of the token vectors we want to look at (so if the dimensionality is 128 then the dimensionality of the subspaces we pay attention to will be 16 as 128/8 = 16), if an element is at drop_probability then we drop it.
        super().__init__() #super() is a function that returns a proxy object of the Parent class (nn.Module). Because it is used within the class, it assumes the parameters it needs so we don't need to rewrite. super().__init__ finds the Parent class init for that object while super().__init__() runs it within this class (hence redefining it within this child class). If we didn't do this then we would still be working with the parent objects/methods within the parent class (this creates a second copy of those attributes and methods within the child class so we are free to manipulate as we wish).
        assert tokenvector_dimension % numberofheads == 0, "Token vector dimension must be divisible by number of heads" #this asserts that the number of heads must divide the token vector dimension (as the dimensionality of the subspaces must be an integer). if the assertion is true the program runs, else it prints the assertion error written.
        self.h = numberofheads
        self.d_k = tokenvector_dimension // numberofheads
        self.qkv = nn.Linear(tokenvector_dimension, 3*tokenvector_dimension, Bias = False) #weight matrix too teach computer Q,K,V
        self.recombine = nn.Linear(tokenvector_dimension, tokenvector_dimension, Bias = False) ##matrix weights to learn how to recombine the heads (acts on concatenation of heads)
        self.drop = nn.Dropout(dropping_probability)
    def forward(self, x, mask):
        B, T, D = x.shape #(x has shape of bias = B x times or context length = T x dimension of token vectors = D)
        q, k, v = self.qkv(x).chunk(3, dim=2) #splitting the weight matrix back up along the D (0=B, 1 = T, 2 = D)
        q_batch = q.view(B,T, self.h, self.d_k).transpose(1,2) #B, H, T, D_k
        k_batch = k.view(B,T, self.h, self.d_k).transpose(1,2)
        v_batch = v.view(B,T, self.h, self.d_k).transpose(1,2)
        similarity_qk = q_batch @ k_batch.transpose(3,4) #so we're getting the similiarity between q and k by taking the dot product or matrix multiply of them (the syntax for that is @). We have to transpose to get the inner dimensions to lay out. This ends up as a BxHxTxT tensor
        
