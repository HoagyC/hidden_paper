import dataclasses
from typing import Optional, List, Tuple, Union

import numpy as np

BASE_DIR = "/root/hoagy-hiddeninfo-sync"

N_CLASSES = 200
N_ATTRIBUTES_RAW = 312 # Number of attr labels for each image in the raw dataset
N_ATTRIBUTES = 109 # Number of attr labels after removing ones used less than 10 times

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclasses.dataclass
class RunRecord:
    epoch: int = -1
    loss: float = float("inf")
    acc: float = 0.0


@dataclasses.dataclass
class Meters:
    loss: AverageMeter = dataclasses.field(default_factory=AverageMeter)
    label_acc: AverageMeter = dataclasses.field(default_factory=AverageMeter)
    concept_acc: AverageMeter = dataclasses.field(default_factory=AverageMeter)


@dataclasses.dataclass
class Experiment:
    tag: str = "basic"
    exp: str = "multimodel"
    seed: int = 1

    # Data
    base_dir: str = BASE_DIR
    log_dir: str = "out"
    data_dir: str = "CUB_masked_class"
    image_dir: str = "images"
    save_step: int = 100
    load: Optional[str] = None # Load a model from a previous run, gives the location of the .pth file

    # Model
    model: Union[str, List[str]] = "inception_v3" # Options: "inception_v3", "resnet18", "resnet34", "resnet50", "resnet101"
    pretrained_weight_n: Union[int, List[int]] = 1 # If using a resnet with multiple options for weights, which to use
    multimodel: bool = True
    n_attributes: int = N_ATTRIBUTES # Width of the concept vector, can be greater but not less than N_ATTRIBUTES
    num_classes: int = N_CLASSES
    expand_dim: int = 500
    use_relu: bool = False
    pretrained: bool = True
    use_aux: bool = True # Whether to use the auxiliary loss from aux_logits
    post_model_dropout: Optional[float] = None

    min_lr = 1e-04
    lr_decay_size: float = 0.1
    aux_loss_ratio: float = 0.4

    # Training
    epochs: List[int] = dataclasses.field(default_factory=list)
    optimizer: str = "SGD"
    scheduler_step: int = 1000
    attr_loss_weight: Union[float, List[float]] = 1.0 # If a list, must be same length as n_models
    class_loss_weight: Union[float, List[float]] = 1.0 # If a list, must be same length as n_models
    lr: float = 1e-03
    weight_decay: float = 2e-4
    attr_sparsity: int = 1 # Masks out the attributes of examples with likelihood 1 - 1/attr_sparsity
    class_sparsity: int = 1 # Masks out the attributes of all examples of a certain class with likelihood 1 - 1/class_sparsity
    batch_size: int = 64
    force_deterministic: bool = True
    use_averaging: bool = False

    freeze: bool = False
    weighted_loss: str = "multiple" # Options: "", "multiple"
    uncertain_labels: bool = False

    # Multi-model specific parameters
    shuffle_models: bool = False
    freeze_post_models: bool = False
    freeze_pre_models: bool = False
    reset_pre_models: bool = False
    reset_post_models: bool = False
    n_models: int = 1
    do_sep_train: bool = True # For basic multimodel runs, whether to train the models separately first
    report_cross_accuracies: bool = False # Whether to report the cross accuracies of the models
    shuffle: bool = True # Whether to shuffle the data while training

    use_test: bool = False # Whether to also check test accuracy as we go
    use_pre_dropout: Union[bool, List[bool]] = True # Whether to include the dropout function in inceptionv3 pre model

    n_alternating: int = 1 # How many times to alternate between pre and post models
    freeze_first: str = "pre" # Whether to freeze the first pre or post model, "pre" or "post"
    alternating_reset: bool = True 
    alternating_epochs: List[int] = dataclasses.field(default_factory=list)
    run_save_path: str = "" # Used to pass path around, don't set manually

    # TTI
    tti_int: int = 10 # run TTI every _ epochs, set to 0 to disable
    tti_model_dir: str = ""
    tti_eval_data: str = "test"  # whether to use test or val data
    flat_intervene: bool = True
    intervene_vals: Tuple[float, float] = (-3, 3) # If intervening with a flat value, what is it?
    n_trials: int = 1 # How many times to run TTI
    n_groups: int = 28  # n. groups of attributes (28 enums with ~10 options = 312 attrs)
    data_dir_raw: str = "CUB_processed"  # directory to the raw data
    feature_group_results: bool = False  # whether to print out performance of individual atttributes
    replace_class: bool = True # Whether to correct with class-averaged or instance level values
    multimodel_type: str = "separate" # Will run them separately if "separate", or will average the logits if "mixture"

@dataclasses.dataclass
class TTI_Output:
    coef: float
    sparsity: int
    result: List[Tuple[int, float]]
    model_name: Optional[str] = None

@dataclasses.dataclass
class Eval_Meter:
    class_accs: List[AverageMeter]


@dataclasses.dataclass
class Eval_Meter_Acc(Eval_Meter):
    attr_acc_tot: AverageMeter = dataclasses.field(default_factory=AverageMeter)
    attr_accs: List[AverageMeter] = dataclasses.field(default_factory=lambda: [])


@dataclasses.dataclass
class Eval_Output:
    class_labels: np.ndarray
    topk_classes: np.ndarray
    class_logits: np.ndarray
    attr_true_labels: np.ndarray
    attr_pred_outputs: np.ndarray
    attr_pred_sigmoids: np.ndarray
    wrong_idx: np.ndarray