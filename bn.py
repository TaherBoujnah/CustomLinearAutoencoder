import torch
import torch.nn as nn

class MeinLayer(nn.Module):
    def __init__(self,input_features,output_features):
        super(MeinLayer, self).__init__()
        self.weight =nn.Parameter(torch.rand(output_features, input_features))
        self.bias = nn.Parameter(torch.rand(output_features))

    def forward(self,X):
        return torch.mm(X,self.weight.t())+self.bias

class Encoder(nn.Module):
    def __init__(self,input_features,output_features):
        super(Encoder,self).__init__()
        self.layer =MeinLayer(input_features,output_features)
    def forward(self,X):
        return self.layer(X)

class Decoder(nn.Module):
    def __init__(self, input_features, output_features):
        super(Decoder, self).__init__()
        self.layer = MeinLayer(input_features, output_features)

    def forward(self, X):
        return self.layer(X)  


if __name__ == "__main__":
    input_features = 4  
    hidden_features = 2  
    output_features = input_features  

    
    encoder = Encoder(input_features, hidden_features)
    decoder = Decoder(hidden_features, output_features)

    
    X = torch.rand(10, input_features)

    
    encoded = encoder(X)  
    reconstructed = decoder(encoded)  

    print("Original Input:")
    print(X)
    print("Encoded Output:")
    print(encoded)
    print("Reconstructed Output:")
    print(reconstructed)