import torch

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.envs import TSPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.utils.trainer import RL4COTrainer


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RL4CO env based on TorchRL
    env = TSPEnv(generator_params=dict(num_loc=20))

    # Model: default is AM with REINFORCE and greedy rollout baseline
    # check out `RL4COLitModule` and `REINFORCE` for more details
    model = AttentionModel(
        env,
        baseline="rollout",
        train_data_size=100_000,  # really small size for demo
        val_data_size=10_000,
    )

    # Example callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # save to checkpoints/
        filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
        save_top_k=1,  # save only the best model
        save_last=True,  # save the last model
        monitor="val/reward",  # monitor validation reward
        mode="max",
    )  # maximize validation reward
    rich_model_summary = RichModelSummary(max_depth=3)  # model summary callback
    callbacks = [checkpoint_callback, rich_model_summary]

    # Logger
    logger = WandbLogger(project="rl4co", name="tsp")
    # logger = None # uncomment this line if you don't want logging

    # Main trainer configuration
    trainer = RL4COTrainer(
        max_epochs=2,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
    )

    # Main training loop
    trainer.fit(model)

    # Greedy rollouts over trained model
    # note: modify this to load your own data instead!
    td_init = env.reset(batch_size=[16]).to(device)
    policy = model.policy.to(device)
    out = policy(td_init.clone(), env, phase="test", decode_type="greedy")

    # Print results
    print(f"Tour lengths: {[f'{-r.item():.3f}' for r in out['reward']]}")
    print(f"Avg tour length: {-torch.mean(out['reward']).item():.3f}")


if __name__ == "__main__":
    main()
