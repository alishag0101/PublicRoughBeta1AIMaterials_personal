import torch
from pathlib import Path
file_path = Path("shakespeare.txt") #defines a reference variable for the shakespeare.txt file. 
text = file_path.read_text(encoding="utf-8")  #method called read_text which reads the file_path and encodes it using the utf-8 encoding. this encoding of file_path is saved under text. 
vocab = sorted(set(text)) #set generates a new list of all the unique characters in text and sorted sorts them. this sorted set of characters is saved under vocab.
vocab_size = len(vocab) #saves length of the vocab list under vocab_size
character_id = {ch:i for i, ch in enumerate{vocab}} #enumerate takes vocab and outputs a list of tuples of the form (index number, index element), sorted by the first dimension. This is "i,ch". Generally, character_id is a dictiionary to go between index and character
number_id = {i:ch for i,ch in enumerate(vocab)} #reverse dictionary from above
encode = lambda s: [character_id(c) for c in s] #define function named "lambda" (which is a general place holder function and keeps from having to explicitly having to define function). Lambda takes in "s" and for every character in "s" it translates using character_id. In the end, the result is a list of numbers.                        
decode = lambda k: "".join(number_id(i) for i in k) #same as above except "".join concatenates together characters into one string
data = torch.tensor(encode(text),dtype=torch.long) #we want to call the function from class torch and pass the two parameters (neither of which are objects of torch so we can't tensor into a bound method))
split_index = int(0.9*len(data)) #this is the index value at which we want to go from training to validation data
train_data = data[:split] #generally tensor[start:end] saves everything from the start index to end index. if start is not specific then python starts read at 0. if end is not specified it ends read at last index.
val_data = data[split:]
def get_batch(split : str, context_length : int = 128, batch_size : int = 32, device : str = "cuda"): #batch refers to a batch of tokens. this a function to generate a batch of tokens. the first parameter determines if we're sampling from the training or validation data. the second parameter refers to length of each indivual token. the third parameters refers to how many tokens we want to pick. the fourth is where on our computer we want to save the tokens. 
    sample = train_data if split=="train" else val_data
    random_int = torch.randint(0, len(sample) - context_length - 1, batch_size) #the goal here is to pick 32 (or batch_size many) starting indices. so this line is defined as 1 by 32 tensor with values between 0 and len(sample) - context_length - 1. We want these starting indices such that a token of context length can be created starting at these indeces AND 1 + these indeces (ie the target or next index for the next token). One d in randomint is a list of 32 numbers
    x = torch.stack([sample[d:d+context_length] for d in random_int]) #stacks the tokens into a 32 tensor like above
    y = torch.stack([sample[d+1:d+context_length+1] for d in random_int])
    return x.to(device), y.to(device)

