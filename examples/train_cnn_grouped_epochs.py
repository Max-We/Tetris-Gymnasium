""" Script to train a DQN agent on Tetris environment using CNN architecture.

The script is a modified version of the [CleanRL's](https://github.com/vwxyzjn/cleanrl) DQN implementation.

docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
"""
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import (
    FeatureVectorObservation,
    RgbObservation,
)


# Evaluation
def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
                )
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


# Training
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tetris_gymnasium_grouped"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    # env_id: str = "BreakoutNoFrameskip-v4"
    env_id: str = "tetris_gymnasium/Tetris"
    """the id of the environment"""
    total_timesteps: int = 60000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 30000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network"""
    batch_size: int = 1
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 1e-3
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 3000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""
    total_epochs: int = 3000
    """the total epochs to train the model"""
    num_decay_epochs: int = 2000
    """the number of epochs to decay the epsilon"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", gravity=False)
            env = GroupedActionsObservations(
                env, observation_wrappers=[FeatureVectorObservation(env)]
            )
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            env = GroupedActionsObservations(
                env, observation_wrappers=[FeatureVectorObservation(env)]
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape[-1]), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._create_weights()


    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    # Env name
    greek_letters = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]
    run_name = f"{args.exp_name}/{random.choice(greek_letters)}_{random.choice(greek_letters)}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        # Log environment code
        run.log_code(
            os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "../tetris_gymnasium"
                )
            )
        )
    writer = SummaryWriter(f"runs_manual/idk")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed)
    action_mask = info["action_mask"][0]

    epoch = 0
    global_step = 0
    epoch_lines_cleared = 0
    while epoch < args.total_epochs:
        # ALGO LOGIC: put action logic here
        epsilon = args.end_e + (max(args.num_decay_epochs - epoch, 0) * (
                args.start_e - args.end_e) / args.num_decay_epochs)

        if random.random() < epsilon:
            actions = np.array(
                [
                    np.random.choice(np.where(action_mask == 1)[0])
                    for env_idx in range(envs.num_envs)
                ]
            )
        else:
            # Normalization by dividing with piece count
            q_values = torch.ones((1, envs.single_action_space.n, 1), dtype=torch.float) * -np.inf
            q_values[:,action_mask == 1,:] = q_network(torch.Tensor(obs[:,action_mask == 1,:]).to(device))
            actions = torch.argmax(q_values, dim=1)[0].cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        action_mask = infos["action_mask"][0]
        epoch_lines_cleared += infos["lines_cleared"][0]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"epoch={epoch}, episodic_return={info['episode']['r']}, episodic_len={info['episode']['l']}, episodic_lines={epoch_lines_cleared}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_lines", epoch_lines_cleared, global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )
                    epoch += 1
                    epoch_lines_cleared = 0

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if rb.size() > args.buffer_size / 10:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (
                        1 - data.dones.flatten()
                )
            old_val = (
                q_network(data.observations)
                .squeeze(-1)
                .gather(1, data.actions)
                .squeeze()
            )
            loss = F.mse_loss(td_target, old_val)

            if global_step % 1 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar(
                    "losses/q_values", old_val.mean().item(), global_step
                )
                writer.add_scalar("losses/target_q", td_target.item(), global_step)
                writer.add_scalar("losses/old_q", old_val.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                writer.add_scalar(
                    "schedule/epsilon",
                    epsilon,
                    global_step,
                )

            print("Loss", loss.item(), td_target.item(), old_val.item())
            # print("Target, Q-Values:", td_target.mean(), old_val.mean())

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        for target_network_param, q_network_param in zip(
            target_network.parameters(), q_network.parameters()
        ):
            target_network_param.data.copy_(
                args.tau * q_network_param.data
                + (1.0 - args.tau) * target_network_param.data
            )

        global_step += 1

    envs.close()
    writer.close()
