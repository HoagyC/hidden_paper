"""
Taken from yewsiang/ConceptBottlenecks
"""
import os
from typing import Optional, List, Tuple, Iterable
from abc import ABC, abstractmethod

import torch
from torch import nn

from model_templates import MLP, inception_v3, resnet50_model, resnet101_model, resnet34_model, resnet18_model
from base_models.fully_connected import FC
from classes import Experiment
from dataset import find_class_imbalance

# Create loss criteria for each attribute, upweighting the less common ones
def make_weighted_criteria(args):
    attr_criterion = []
    train_data_path = os.path.join(args.base_dir, args.data_dir, "train.pkl")
    imbalance = find_class_imbalance(train_data_path, True) # assume args.weighted loss is always "multiple" if not ""
    for ratio in imbalance[:args.n_attributes]:
        attr_criterion.append(
            torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda())
        )
    return attr_criterion


def unsqueeze(x: Optional[torch.Tensor], dim: int = 0):
    if x is None:
        return None
    return x.unsqueeze(dim)

def squeeze(x: Optional[torch.Tensor], dim: int = 0):
    if x is None:
        return None
    return x.squeeze(dim)

# Basic model for predicting attributes from images
def ModelXtoC(args: Experiment, use_dropout: Optional[bool] = None, model_str: Optional[str] = None, weight_n: int = 1) -> nn.Module:
    """
    Model for predicting attributes from images.
    Takes in an image and outputs a list of outputs for each attribute,
    where the output is a vector of size (batch_size, 1).
    """
    if model_str is None:
        assert isinstance(args.model, str)
        model_str = args.model

    if use_dropout is None:
        assert isinstance(args.use_pre_dropout, bool)
        use_dropout = args.use_pre_dropout

    if model_str == "resnet50":
        print(f"Using ResNet50 with dropout: {use_dropout}, weight_n: {weight_n}")
        return resnet50_model(args, weight_n=weight_n)
    
    elif model_str == "resnet101":
        print(f"Using ResNet101 with dropout: {use_dropout}, weight_n: {weight_n}")
        return resnet101_model(args, weight_n=weight_n)

    elif model_str == "resnet34":
        print(f"Using ResNet34 with dropout: {use_dropout}, weight_n: {weight_n}")
        if weight_n != 1:
            print("Note: there is only one set of weights for ResNet34, so weight_n is ignored.")
        return resnet34_model(args)

    elif model_str == "resnet18":
        print(f"Using ResNet18 with dropout: {use_dropout}, weight_n: {weight_n}")
        if weight_n != 1:
            print("Note: there is only one set of weights for ResNet18, so weight_n is ignored.")
        return resnet18_model(args)
    
    elif model_str == "inception_v3":
        print(f"Using inception_v3 with dropout: {use_dropout}, weight_n: {weight_n}")
        return inception_v3(
            pretrained=args.pretrained,
            freeze=False,
            aux_logits=args.use_aux,
            num_classes=args.num_classes,
            n_attributes=args.n_attributes,
            expand_dim=args.expand_dim,
            use_dropout=use_dropout,
        )
    else:
        raise ValueError(f"Model {model_str} not supported, use resnet{{18, 34, 50, 101}} or inception_v3")

# Basic model for predicting classes from attributes
def ModelCtoY(args: Experiment) -> nn.Module:
    model = MLP(
        input_dim=args.n_attributes, num_classes=args.num_classes, expand_dim=args.expand_dim, post_model_dropout=args.post_model_dropout
    )
    return model


# Base class that all models inherit from
class CUB_Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.train_mode: str
        self.args: Experiment

    @abstractmethod
    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through=None) -> \
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None, None, None

    @abstractmethod
    def generate_loss(self, 
        attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        attr_labels: Optional[torch.Tensor], # batch_size x n_attributes
        class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        class_labels: Optional[torch.Tensor], # batch_size
        aux_class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        aux_attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        mask: torch.Tensor,):
        pass


class CUB_Multimodel(CUB_Model):
    def __init__(self) -> None:
        super().__init__()
        self.pre_models: nn.ModuleList
        self.post_models: nn.ModuleList

    @abstractmethod
    def reset_pre_models(self) -> None:
        pass

    @abstractmethod
    def reset_post_models(self):
        pass

    def init_loss_weights(self, args: Experiment) -> None:
        self.class_loss_weights: List[float]
        if isinstance(args.class_loss_weight, list):
            assert len(args.class_loss_weight) == args.n_models
            self.class_loss_weights = args.class_loss_weight
        else:
            self.class_loss_weights = [args.class_loss_weight] * args.n_models
        
        self.attr_loss_weights: List[float]
        if isinstance(args.attr_loss_weight, list):
            assert len(args.attr_loss_weight) == args.n_models
            self.attr_loss_weights = args.attr_loss_weight
        else:
            self.attr_loss_weights = [args.attr_loss_weight] * args.n_models

        self.dropouts: List[bool]
        if isinstance(args.use_pre_dropout, list):
            assert len(args.use_pre_dropout) == args.n_models
            self.dropouts = args.use_pre_dropout
        else:   
            self.dropouts = [args.use_pre_dropout] * args.n_models
        
        print(f"Class loss weights: {self.class_loss_weights}, attr loss weights: {self.attr_loss_weights}")

class Multimodel(CUB_Multimodel):
    def __init__(self, args: Experiment):
        super().__init__()
        self.args = args
        self.pre_models: nn.ModuleList
        self.post_models: nn.ModuleList
        self.init_loss_weights(args)
        self.reset_pre_models()
        self.reset_post_models()
        self.train_mode: str = "separate"
        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]
        self.av_diff_same: List[torch.Tensor] = []
        self.av_diff_switch: List[torch.Tensor] = []

    def reset_pre_models(self) -> None:
        if isinstance(self.args.model, list):
            assert len(self.args.model) == self.args.n_models
        else:
            self.args.model = [self.args.model] * self.args.n_models

        if isinstance(self.args.pretrained_weight_n, list):
            assert len(self.args.pretrained_weight_n) == self.args.n_models
        else:
            self.args.pretrained_weight_n = [self.args.pretrained_weight_n] * self.args.n_models
    
        pre_models_list = [
            ModelXtoC(self.args, use_dropout=self.dropouts[i], model_str=self.args.model[i], weight_n=self.args.pretrained_weight_n[i])
            for i in range(self.args.n_models)
        ]
        self.pre_models = nn.ModuleList(pre_models_list)

    def reset_post_models(self) -> None:
        post_models_list = [
            nn.Sequential(nn.Sigmoid(), ModelCtoY(self.args))
            for _ in range(self.args.n_models)
        ]

        self.post_models = nn.ModuleList(post_models_list)

    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through=None) -> \
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        attr_preds = []
        aux_attr_preds = []
        class_preds = []
        aux_class_preds = []
        post_model_indices: Iterable
        # Train each pre model with its own post model
        if self.train_mode == "separate":
            post_model_indices = range(self.args.n_models)
        
        # Randomly shuffle which post model is used for each pre model
        elif self.train_mode == "shuffle":
            post_model_indices = torch.randperm(self.args.n_models)
            
        else:
            raise ValueError(f"Invalid train mode {self.train_mode}")
        
        # If we want to make sure that we're using different training orders then we'll have to split the batch up between the different models
        batch_ndxs = self.make_batch_ndxs(inputs.shape[0])

        for i, j in enumerate(post_model_indices):
            attr_pred, aux_attr_pred = self.pre_models[i](inputs[batch_ndxs[i]])
            class_pred = self.post_models[j](attr_pred)
            
            attr_preds.append(attr_pred)
            class_preds.append(class_pred)
            if self.args.use_aux:
                aux_class_pred = self.post_models[j](aux_attr_pred)
                aux_attr_preds.append(aux_attr_pred)
                aux_class_preds.append(aux_class_pred)

        if self.args.use_aux:
            return torch.stack(attr_preds), torch.stack(aux_attr_preds), torch.stack(class_preds), torch.stack(aux_class_preds)
        else:
            return torch.stack(attr_preds), None, torch.stack(class_preds), None
    
    
    def generate_loss(
        self, 
        attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        attr_labels: Optional[torch.Tensor], # batch_size x n_attributes
        class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        class_labels: Optional[torch.Tensor], # batch_size
        aux_class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        aux_attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        mask: torch.Tensor,
    ):
        total_class_loss = 0.
        total_attr_loss = 0.
        
        assert attr_preds is not None and attr_labels is not None
        assert class_preds is not None and class_labels is not None

        assert len(attr_preds) == len(class_preds) == len(self.pre_models)

        batch_ndxs = self.make_batch_ndxs(attr_labels.shape[0], mask)
        
        for ndx in range(len(self.pre_models)):
            class_loss = self.criterion(class_preds[ndx], class_labels[batch_ndxs[ndx]])
            if self.args.use_aux:
                assert aux_class_preds is not None
                aux_class_loss = self.criterion(aux_class_preds[ndx], class_labels[batch_ndxs[ndx]])
            else:
                aux_class_loss = 0.
            class_loss += aux_class_loss * self.args.aux_loss_ratio
            total_class_loss += class_loss * self.class_loss_weights[ndx]
            
            attr_loss = 0.
            for i in range(len(self.attr_criterion)):
                in_batch_mask = [x % (self.args.batch_size // self.args.n_models) for x in batch_ndxs[ndx]]
                attr_loss += self.attr_criterion[i](
                    attr_preds[ndx, in_batch_mask, i], attr_labels[batch_ndxs[ndx], i] # Masking attr losses
                )
                if self.args.use_aux:
                    assert aux_attr_preds is not None
                    aux_attr_loss = self.attr_criterion[i](
                        aux_attr_preds[ndx, in_batch_mask, i], attr_labels[batch_ndxs[ndx], i] # Masking attr losses
                    )
                else:
                    aux_attr_loss = 0.
                attr_loss += aux_attr_loss * self.args.aux_loss_ratio
            
            attr_loss /= len(self.attr_criterion)
            total_attr_loss += attr_loss * self.attr_loss_weights[ndx]
    
        loss = total_attr_loss + total_class_loss
        return loss
    

class SequentialModel(CUB_Model):
    def __init__(self, args: Experiment, train_mode: str) -> None:
        super().__init__()
        self.args = args
        self.first_model = ModelXtoC(self.args)
        self.second_model = nn.Sequential(nn.Sigmoid(), ModelCtoY(self.args))
        self.criterion = nn.CrossEntropyLoss()
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]

        assert not isinstance(args.attr_loss_weight, list)
        self.attr_loss_weight = args.attr_loss_weight
        self.train_mode = train_mode

    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through=None):
        attr_preds, aux_attr_preds = self.first_model(inputs[mask])

        # Attr preds are list of tensors, need to concat them with batch as d0
        # Detach to prevent gradients from flowing back to first model

        if self.train_mode == "CtoY":
            class_preds = self.second_model(attr_preds.detach())
            aux_class_preds = self.second_model(aux_attr_preds.detach())
            
        else:
            class_preds, aux_class_preds = None, None
        
        return unsqueeze(attr_preds), unsqueeze(aux_attr_preds), unsqueeze(class_preds), unsqueeze(aux_class_preds)

    def generate_loss(
        self,
        attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        attr_labels: Optional[torch.Tensor], # batch_size x n_attributes
        class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        class_labels: Optional[torch.Tensor], # batch_size
        aux_class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        aux_attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        mask: torch.Tensor, # batch_size
    ):
        attr_preds = squeeze(attr_preds) # Removing the n_models dimension
        aux_attr_preds = squeeze(aux_attr_preds)
        class_preds = squeeze(class_preds)
        aux_class_preds = squeeze(aux_class_preds)

        attr_loss = 0.
        if self.train_mode == "XtoC":
            assert attr_preds is not None and attr_labels is not None and aux_attr_preds is not None
            
            for i in range(len(self.attr_criterion)):
                attr_loss += self.attr_criterion[i](
                    attr_preds[mask, i], attr_labels[mask, i]
                )
        
            attr_loss /= len(self.attr_criterion)

        if self.train_mode == "CtoY":
            assert class_preds is not None and class_labels is not None and aux_class_preds is not None

            class_loss = self.criterion(class_preds, class_labels[mask])
        else:
            class_loss = 0

        loss = (attr_loss * self.attr_loss_weight) + class_loss
        return loss

class JointModel(CUB_Model):
    def __init__(self, args: Experiment) -> None:
        super().__init__()
        self.args = args
        self.first_model = ModelXtoC(self.args)
        self.second_model = nn.Sequential(nn.Sigmoid(), ModelCtoY(self.args))

        self.criterion = nn.CrossEntropyLoss()
    
        if self.args.weighted_loss:
            self.attr_criterion = make_weighted_criteria(args)
        else:
            self.attr_criterion = [torch.nn.CrossEntropyLoss() for _ in range(args.n_attributes)]

        assert not isinstance(args.attr_loss_weight, list)
        self.attr_loss_weight = args.attr_loss_weight
        self.train_mode = "joint"
        print(self.second_model)

    def generate_predictions(self, inputs: torch.Tensor, attr_labels: torch.Tensor, mask: torch.Tensor, straight_through=None):
        attr_preds, aux_attr_preds = self.first_model(inputs)
    
        class_preds = self.second_model(attr_preds)
        aux_class_preds = self.second_model(aux_attr_preds)

        # Add extra dimension for consistency with multimodels
        return unsqueeze(attr_preds), unsqueeze(aux_attr_preds), unsqueeze(class_preds), unsqueeze(aux_class_preds)

    def generate_loss(
        self,
        attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        attr_labels: Optional[torch.Tensor], # batch_size x n_attributes
        class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        class_labels: Optional[torch.Tensor], # batch_size
        aux_class_preds: Optional[torch.Tensor], # n_models x batch_size x n_classes (but probably None)
        aux_attr_preds: Optional[torch.Tensor], # n_models x batch_size x n_attributes
        mask: torch.Tensor, # batch_size
    ):
        attr_preds = squeeze(attr_preds) # Removing the n_models dimension
        aux_attr_preds = squeeze(aux_attr_preds)
        class_preds = squeeze(class_preds)
        aux_class_preds = squeeze(aux_class_preds)

        assert attr_preds is not None and attr_labels is not None and aux_attr_preds is not None
        assert class_preds is not None and class_labels is not None and aux_class_preds is not None

        class_loss = self.criterion(class_preds, class_labels)
        aux_class_loss = self.criterion(aux_class_preds, class_labels)
        class_loss += aux_class_loss * self.args.aux_loss_ratio
        
        attr_loss = 0.
        for i in range(len(self.attr_criterion)):
            attr_loss += self.attr_criterion[i](
                attr_preds[mask, i], attr_labels[mask, i] # Masking attr losses
            )

            aux_attr_loss = self.attr_criterion[i](
                aux_attr_preds[mask, i], attr_labels[mask, i] # Masking attr losses
            )
            attr_loss += aux_attr_loss * self.args.aux_loss_ratio

        
        attr_loss /= len(self.attr_criterion)
        loss = (attr_loss * self.attr_loss_weight) + class_loss
        return loss

