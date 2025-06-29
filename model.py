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
        similarity_qk = q_batch @ k_batch.transpose(3,4) #so we're getting the similiarity between q and k by taking the dot product or matrix multiply of them (the syntax for that is @). We have to transpose to get the inner dimensions to lay out. This ends up as a BxHxTxT tensors
        probabilitydropped = similarity_qk.masked_fill(mask == 0, -1e4) #masked_fill is a PyTorch method that drops the probability (-1e4) for a_{ij} in the source matrix (similiarityqk) when the entry of mask (a matrix of 0s and 1s) in mask_ij is equal to the condition in the first parameter (so when mask_{ij} = 0 then the weight attatched to how much attention q #i pays to k #j drops very low).
        summingoverkeys = probabilitydropped.softmax(dim = -1) #this softmax basically turns a_ij = e^{a_ij}/sum(e^{ik}). so for every quiery i, it first calculates the sum of e^{a_ij} for all j (so for every entry in the row). Then divides e^{a_ij} by it for every a_{ij} in the ith row. Then the ith row sums to one and each row becomes a mini probability distribution for how much attention the quiery of that row pays to the keys.
        finalqk = self.drop(summingoverkeys) #this just drops some dropping_probability percentage of entries to zero to avoid overfitting (it creates the mask matrix where the dropping probability or prob of entry being 0 is dropping probability). there are two diff masks tho - this is the dropping mask and the one in line 20 was the padding mask.
        valueextraction = finalqk @ v_batch #This will have B,H, T, D_k because we have B,H,T,T @ B,H,T,D_k so we only have to look at T,T @ T,D_k which is a T,D_k submatrix so the total matix will be B, H, T, D_k
        final = valueextraction.transpose(1,2).contingious().view(B, T, D) #we want to contract the heads and dimension per heads dimensions into one so we bring them both to end (spots 3 and 4 by the transpose). Then we apply contigious as a contigious tensor is required for view to work and then we change the shape of tensor with view. 
        return self.recombine(final) #applies matrix weights to final so we can update it and learn the best way to recombin the heads 
class TransformerBlock(nn.Module):
    def __init__(self, tokenvector_dimension: int, numberofheads: int, dropping_probability: float)
        super().__init__()
        self.layernorm1 = nn.LayerNorm(tokenvector_dimension) #this just makes the embedding vector numbers for each indiviual token have zero mean and a spread of 1
        self.attention = MultiHeadSelfAttention(tokenvector_dimension, numberofheads, dropping_probability)
        self.layernorm2 = nn.LayerNorm(tokenvector_dimension)
        self.ff = nn.Sequential( #the feed forward network is a sequence of linear and non linear operators that adjust the numbers inside of each indivual token vector embedding (so that we can back prop and adjust weights later), independent of context (so no attention).
            nn.Linear(tokenvector_dimension, 4 * tokenvector_dimension), #a widening matrix, linear transformation 1 
            nn.GELU(), #a non linear transformation
            nn.Linear(tokenvector_dimension, tokenvector_dimension), #a contracting matrix, linear transformation 2
            nn.Dropout
        )
    def forward(self, x, mask):
        x = x + self.attention(self.layernorm1(x), mask) #here we are treating self.attention as a function because it is both object and function due to inheriting the properites of nn.Module (super()) which inheritly allows any object to be called as a function defined by applying the forward function of whatever class the object belongs to. 
        x = x + self.ff(self.layernorm2(x))
        return x 


