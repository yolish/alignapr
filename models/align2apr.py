import torch
import torch.nn as nn

def align_to_apr(imgs, encoder, mapper):
    with torch.no_grad():
        features = encoder(imgs)
    res = mapper(features)
    return res

class Mapper(nn.Module):
    def __init__(self, config):
        super(Mapper).__init()
        mlp_dims = [config["input_dim"], *config["mlp_dim"]]
        self.mlp = self.get_mlp(dims=mlp_dims,
                            dropout=config["mlp_dropout"])
        mlp_out_dim = mlp_dims[-1]
        ori_repr = config["orientation_representation"]
        q_dim = 4
        if ori_repr == "quat":
            q_dim = 4
        elif ori_repr == "6d":
            q_dim = 6
        else: 
            raise Exception("{} not supported for orientation representation".format(ori_repr))
        self.fc_x = nn.Linear(mlp_out_dim, 3)
        self.fc_q = nn.Linear(mlp_out_dim, q_dim)

    @staticmethod
    def get_mlp(dims, dropout):
        layers = []
        for i in range(0, len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropput(p=dropout))
        return nn.Sequential(layers)

    def forward(self, data):
        features = data["features"] # B x D
        features = self.mlp(v)
        x = self.fc_x(features)
        q = self.fc_q(features)
        return {"features":features,
                "x": x,
                "q": q}

 