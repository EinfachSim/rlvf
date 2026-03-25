import torch

class HyperNetwork(torch.nn.Module):
    def __init__(self, num_layers, layer_types, dims, emb_l_dim=32, emb_t_dim=32, rank=8, hidden_dim = 512):
        super().__init__()

        self.num_layers = num_layers
        self.num_types = len(layer_types)
        self.layer_types = layer_types

        #Precomputed layer and type batches
        layer_indices = torch.arange(self.num_layers)
        type_indices = torch.arange(self.num_types)

        l, t = torch.meshgrid(layer_indices, type_indices, indexing="ij")

        self.register_buffer("l", l.flatten())
        self.register_buffer("t", t.flatten())

        #Network setup

        #Embeddings
        self.layer_emb = torch.nn.Embedding(self.num_layers, emb_l_dim)
        self.type_emb = torch.nn.Embedding(self.num_types, emb_t_dim)

        #Mixer on concat TODO
        self.mixer = None

        #TODO
        self.mlp1 = None
        self.mlp2 = None
        self.mlp3 = None

        #A and B
        self.A = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(torch.empty(num_layers, rank, dims[k][1])) for k in layer_types
            }
        )
        #A is initialized as kaiming
        for _, A in self.A.items():
            torch.nn.init.kaiming_uniform_(A, a=0, mode="fan_in")
        
        #B is initialized as zero
        self.B = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(torch.zeros(num_layers, rank, dims[k][0])) for k in layer_types
            }
        )

        #Simple linear layer from hidden_dim to rank
        self.head = torch.nn.Linear(hidden_dim, rank)
    
    def forward(self, x):
        batch_size = x.shape[0]

        x = x.unsqueeze(1).expand(-1, self.num_layers*self.num_types, -1)

        l = self.l.unsqueeze(0).expand(batch_size, -1)
        t = self.t.unsqueeze(0).expand(batch_size, -1)
        
        #Embeddings
        emb_l = self.layer_emb(l)
        emb_t = self.type_emb(t)
        x = torch.concat([x, emb_l, emb_t], dim=-1)
        print(x.shape)
        
        #x = self.mixer(x)

        #MLP
        #x = self.mlp1(x)


hn = HyperNetwork(3, ["q", "v"], dims={"q": [10, 20], "v": [20, 10]})

hn(torch.arange(3).unsqueeze(0).expand(10, -1))