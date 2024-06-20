import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.loss import RegLoss
from torch.nn import functional as F

def RotationBasedAttention(theta_a, theta_b, weight=None):
    cos_a, cos_b = torch.cos(theta_a), torch.cos(theta_b)
    sin_a, sin_b = torch.sin(theta_a), torch.sin(theta_b)
    if weight is not None:
        cos_a, sin_a = cos_a * weight, sin_a * weight
    return torch.sigmoid(cos_a @ cos_b.transpose(-2, -1) + sin_a @ sin_b.transpose(-2, -1)) # cos a-b = cosa cosb + sin a sin b

class SelfAttentiveRotation(nn.Module):
    def __init__(self, config, input_shape, output_shape) -> None:
        super().__init__()
        self.input_shape, self.output_shape = input_shape, output_shape
        field_num = self.field_num = config.num_field
        self.head_num = config.head_num
        self.initialize_parameters(field_num, input_shape, output_shape, config.drop_rate_att)

    def initialize_parameters(self, field_num, input_shape, output_shape, dropout_prob):
        self.Q, self.K, self.V = [
            nn.Parameter(torch.ones(field_num, input_shape, output_shape))
            for _ in range(3)
        ]
        self.linear_transform = nn.Linear(input_shape, output_shape, bias=False)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.LayerNorm([self.output_shape])
        self.weight = nn.Parameter(torch.ones(1, self.output_shape // self.head_num))
        for param in [self.Q, self.K, self.V, self.linear_transform.weight, self.weight]:
            xavier_normal_(param, gain=1.414)

    def forward(self, theta):
        tensor = theta.reshape(theta.shape[0], -1, self.input_shape)
        queries, keys, values = [torch.einsum('bfd,fdw->bfw', tensor, param) for param in [self.Q, self.K, self.V]]
        head_tensors = [
            torch.stack(torch.split(tensor, [self.output_shape // self.head_num] * self.head_num, dim=-1), dim=1)
            for tensor in [queries, keys, values]
        ]
        queries, keys, values = head_tensors

        scores = RotationBasedAttention(queries, keys, weight=self.weight).transpose(-2, -1)
        values = self.dropout(values)
        output = torch.cat(
            torch.split(
                torch.transpose(values @ scores, -2, -1),
                [1] * self.head_num, dim=1
            ), dim=-1
        ).squeeze(1)
        theta = self.dropout(theta)
        representation = self.norm(output + self.linear_transform(theta))
        return representation

class RFM(ContextRecommender):
    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)

        self.attention_architecture = [config.embedding_size] + [config.hidden_units] * config.att_layers
        config.num_field = self.num_feature_field
        self.saro_layers = nn.Sequential(
            *(SelfAttentiveRotation(config, in_shape, out_shape) 
              for in_shape, out_shape in zip(self.attention_architecture[:-1], self.attention_architecture[1:]))
        )
        
        self.norm1 = nn.LayerNorm([self.attention_architecture[-1]])
        self.norm2 = nn.LayerNorm([self.attention_architecture[-1]])
        
        config.mlp_list = [self.attention_architecture[-1] * config.num_field] + config.mlp_list
        self.ampnet = AmpNet(config)
        self.add_first_order_residual = config.add_first_order_residual
        
        self.projection = nn.Linear(config.embedding_size, self.attention_architecture[-1])
        nn.init.normal_(self.projection.weight, mean=0, std=0.01)
        
        self.apply(self._initialize_weights)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.reg_loss = RegLoss()

    def _initialize_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)

    def forward(self, interaction):
        angular_embeddings = self.concat_embed_input_fields(interaction)
        theta = self.saro_layers(angular_embeddings)
        r, p = torch.cos(theta), torch.sin(theta)
        
        if self.add_first_order_residual:
            r = self.norm1(r + torch.cos(self.projection(angular_embeddings)))
            p = self.norm2(p + torch.sin(self.projection(angular_embeddings)))
        
        logits = self.ampnet((r, p)).squeeze(-1)
        return logits

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))

class Amp(nn.Module):
    def __init__(self, config, input_shape, output_shape, activation=None) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.group_size = config.group_hidden_dimension
        
        # Initialize linear layer
        self.linear = nn.Linear(input_shape, output_shape)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        
        # Initialize dropout layer
        self.dropout = nn.Dropout(p=config.dp_rate_amp)
        
        # Activation function
        self.activation = activation
        
        # Normalization layers, here we use LayerNorm to implement GroupNorm
        self.norm1 = nn.LayerNorm([self.group_size])
        self.norm2 = nn.LayerNorm([self.group_size])
        
    def forward(self, kws):
        real, imag = kws
        real = self.dropout(real).view(real.size(0), -1)
        imag = self.dropout(imag).view(imag.size(0), -1)
        real, imag = self.linear(real), self.linear(imag)
        if self.activation:
            real, imag = self.activation(real), self.activation(imag)
        real = real.view(real.size(0), -1, self.group_size)
        imag = imag.view(imag.size(0), -1, self.group_size)
        real, imag = self.norm1(real), self.norm2(imag)
        return real, imag


class AmpNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.mlp_config_list = config.mlp_list
        mlps = [Amp(config, in_shape, out_shape, nn.ReLU())
                for in_shape, out_shape in zip(self.mlp_config_list[:-1], self.mlp_config_list[1:])]
        
        self.block = nn.Sequential(*mlps)
        
        # Regularization layer
        self.regularization = nn.Linear(self.mlp_config_list[-1], 1)
        nn.init.xavier_normal_(self.regularization.weight)

    def forward(self, feature):
        real, imag = feature
        real, imag = self.block((real, imag))
        real = real.view(real.size(0), -1)
        imag = imag.view(imag.size(0), -1)
        real, imag = self.regularization(real), self.regularization(imag)
        return real + imag