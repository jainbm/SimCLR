import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                            "mobilev2": models.mobilenet_v2(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        if base_model == 'mobilev2':
            dim_mlp = self.backbone.classifier[1].in_features
        else:
            dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        if base_model == 'mobilev2':
            self.backbone.classifier[1] = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.classifier[1])
        else:
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:

            model = self.resnet_dict[model_name]
            print(f'{model_name} Backbone Selected')
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
