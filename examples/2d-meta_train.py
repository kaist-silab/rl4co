from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.trainer import RL4COTrainer
from rl4co.utils.meta_trainer import ReptileCallback

def main():
    # Set device
    device_id = 0

    # RL4CO env based on TorchRL
    env = CVRPEnv(generator_params={'num_loc': 50})

    # Policy: neural network, in this case with encoder-decoder architecture
    # Note that this is adapted the same as POMO did in the original paper
    policy = AttentionModelPolicy(env_name=env.name,
                                  embed_dim=128,
                                  num_encoder_layers=6,
                                  num_heads=8,
                                  normalization="instance",
                                  use_graph_context=False
                                  )

    # RL Model (POMO)
    model = POMO(env,
                 policy,
                 batch_size=64,  # meta_batch_size
                 train_data_size=64 * 50,  # equals to (meta_batch_size) * (gradient decent steps in the inner-loop optimization of meta-learning method)
                 val_data_size=0,
                 optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
                 )

    # Example callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="meta_pomo/checkpoints",  # save to checkpoints/
        filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
        save_top_k=1,  # save only the best model
        save_last=True,  # save the last model
        monitor="val/reward",  # monitor validation reward
        mode="max",  # maximize validation reward
    )
    rich_model_summary = RichModelSummary(max_depth=3)  # model summary callback

    # Meta callbacks
    meta_callback = ReptileCallback(
        num_tasks = 1,  # the number of tasks in a mini-batch, i.e. `B` in the original paper
        alpha = 0.9,  # initial weight of the task model for the outer-loop optimization of reptile
        alpha_decay = 1,  # weight decay of the task model for the outer-loop optimization of reptile. No decay performs better.
        min_size = 20,  # minimum of sampled size in meta tasks (only supported in cross-size generalization)
        max_size= 150,  # maximum of sampled size in meta tasks (only supported in cross-size generalization)
        data_type="size_distribution",  # choose from ["size", "distribution", "size_distribution"]
        sch_bar=0.9,  # for the task scheduler of size setting, where lr_decay_epoch = sch_bar * epochs, i.e. after this epoch, learning rate will decay with a weight 0.1
        print_log=True # whether to print the sampled tasks in each meta iteration
    )
    callbacks = [meta_callback, checkpoint_callback, rich_model_summary]

    # Logger
    logger = WandbLogger(project="rl4co", name=f"{env.name}_pomo_reptile")
    # logger = None # uncomment this line if you don't want logging

    # Adjust your trainer to the number of epochs you want to run
    trainer = RL4COTrainer(
        max_epochs=15000,  # (the number of meta_model updates) * (the number of tasks in a mini-batch)
        callbacks=callbacks,
        accelerator="gpu",
        devices=[device_id],
        logger=logger,
        limit_train_batches=50  # gradient decent steps in the inner-loop optimization of meta-learning method
    )

    # Fit
    trainer.fit(model)


if __name__ == "__main__":
    main()

