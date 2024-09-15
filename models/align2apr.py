import torch
import torch.nn as nn

def align_to_apr(imgs, encoder, mapper):
    with torch.no_grad():
        features = encoder(imgs)
    res = mapper(features)
    return res

def get_encoder(config):
    encoder_name = config["encoder_name"]
    encoder_params = config[encoder_name]
    output_dim = encoder_params["output_dim"]
    if encoder_name == "eigenplaces":
        encoder = torch.hub.load("gmberton/eigenplaces", "get_trained_model", 
                       backbone=encoder_params["backbone"], fc_output_dim=output_dim)
    else:
        raise NotImplementedError("{} is not supported for encoder type".format(encoder_name))
    return encoder, output_dim 

class Mapper(nn.Module):
    def __init__(self, config):
        super(Mapper, self).__init__()
        self.config = config
        self.mlp, mlp_out_dim = self.get_mlp()
        ori_repr = self.config["orientation_representation"]
        q_dim = 4
        if ori_repr == "quat":
            q_dim = 4
        elif ori_repr == "6d":
            q_dim = 6
        else: 
            raise Exception("{} not supported for orientation representation".format(ori_repr))
        self.fc_x = nn.Linear(mlp_out_dim, 3)
        self.fc_q = nn.Linear(mlp_out_dim, q_dim)

    def get_mlp(self):
        dims = [self.config["input_dim"], *self.config["mlp_dims"]]
        mlp_out_dim = dims[-1]
        layers = []
        for i in range(0, len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=self.config["mlp_dropout"]))
        return nn.Sequential(*layers), mlp_out_dim

    def forward(self, features):
        features = self.mlp(features)
        x = self.fc_x(features)
        q = self.fc_q(features)
        p = torch.cat([x,q], dim=1)
        return {"features":features,
                "pose": p}

 