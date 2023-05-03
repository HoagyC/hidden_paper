import dataclasses
from typing import List, Dict

from classes import Experiment

basic_config = Experiment(
    exp="Multimodel",
    multimodel=True,
    epochs = [150],
    batch_size = 32,
    n_models=1,
    data_dir = "CUB_instance_masked"
)

cfg_lists: Dict[str, List[Experiment]] = {}

# To get the sequential baseline for benchmarking the hidden information
sequentials = []
for ndx in range(3):
    new_cfg = dataclasses.replace(
        basic_config,
        exp="MultiSequential",
        tag=f"seq_inst_{ndx}",
        seed=ndx,
    )
    sequentials.append(new_cfg)
cfg_lists["sequentials"] = sequentials

# To do a sweep of certain basic training properties to show how they impact the results
# ones to use: expand_dim, use_dropout, use_aux, diff_order, adding {1, 10} additional blank dims
train_sweep = []
change_list = [
    ('expand_dim', 200),
    ('use_pre_dropout', True),
    ('shuffle', False), # Need to halve epochs if they all see the same examples
    ('use_aux', False),
    ('n_attributes', 110),
    ('n_attributes', 119),
]
for change in change_list:
    new_cfg = dataclasses.replace(
        basic_config,
        tag=f"multi_inst_{change[0]}_{change[1]}",
        **{change[0]: change[1]}
    )
    new_cfg.n_models = 4
    new_cfg.batch_size = 8
    train_sweep.append(new_cfg)
cfg_lists["train_sweep"] = train_sweep

# To do a sweep of different attr_loss_weight values to show how they impact the results
test_attr_loss_weights = []
for attr_loss_weight in [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10]:
    for dropout in [False, True]:
        dropout_str = "dropout" if dropout else "no-dropout"
        new_cfg = dataclasses.replace(
            basic_config,
            tag=f"multi_inst_attr_loss_{attr_loss_weight}_{dropout_str}",
            attr_loss_weight=attr_loss_weight,
            use_pre_dropout=dropout
        )
        test_attr_loss_weights.append(new_cfg)
cfg_lists["loss_weights"] = test_attr_loss_weights

# Doing a sweep of different models to show how they impact the results
test_all_archs = []
for arch in ["resnet18", "resnet34", "resnet50", "resnet101", "inception_v3"]:
    new_cfg = dataclasses.replace(
        basic_config,
        tag=f"multi_inst_{arch}",
        model = arch,
        use_aux = False
    )
    test_all_archs.append(new_cfg)
    if arch in ["resnet50", "resnet101"]:
        new_cfg = dataclasses.replace(
            new_cfg,
            tag=f"multi_inst_{arch}2",
            model = arch,
            pretrained_weight_n = 2
        )
        test_all_archs.append(new_cfg)
cfg_lists["test_all_archs"] = test_all_archs