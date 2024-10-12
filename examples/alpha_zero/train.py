"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

import random
import time
from collections import defaultdict, deque

import gymnasium
import numpy as np

from examples.alpha_zero.agent import MCTSAgent
from examples.alpha_zero.model import PolicyValueNet
from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.observation import SimpleObservationWrapper


class TrainPipeline:
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 10
        self.board_height = 20
        self.n_in_row = 4
        self.env = SimpleObservationWrapper(
            gymnasium.make(
                "tetris_gymnasium/Tetris",
                width=self.board_width,
                height=self.board_height,
                render_mode="rgb_array",
                gravity=True,
            )
        )
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        action_size = self.env.action_space.n
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(
                self.board_width, self.board_height, action_size, model_file=init_model
            )
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(
                self.board_width, self.board_height, action_size
            )
        self.agent = MCTSAgent(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
        )

    def start_self_play(self, is_shown=False):
        """start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        obs, _ = self.env.reset()
        states, mcts_probs, rewards = [], [], []
        while True:
            move, move_probs = self.agent.get_action(self.env, temp=self.temp, return_prob=1)
            # store the data
            states.append(obs)
            mcts_probs.append(move_probs)
            # perform a move
            obs, reward, terminated, truncated, info = self.env.step(move)
            rewards.append(reward)
            if is_shown:
                self.env.render()
            if terminated:
                self.agent.reset_agent()
                if is_shown:
                    print("Game over")
                return zip(states, mcts_probs, rewards)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            play_data = self.start_self_play()
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        z_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                z_batch,
                self.learn_rate * self.lr_multiplier,
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(z_batch) - old_v.flatten()) / np.var(
            np.array(z_batch)
        )
        explained_var_new = 1 - np.var(np.array(z_batch) - new_v.flatten()) / np.var(
            np.array(z_batch)
        )
        print(
            (
                "kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.3f},"
                "explained_var_new:{:.3f}"
            ).format(
                kl,
                self.lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new,
            )
        )
        return loss, entropy

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                start_time = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                end_time = time.time()

                print(f"batch i:{i + 1}, episode_len:{self.episode_len}")

                execution_time = end_time - start_time
                print(f"collect_selfplay_data took {execution_time:.4f} seconds")
                print("SPS", self.episode_len / execution_time)

                print(f"batch i:{i + 1}, episode_len:{self.episode_len}")
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    print(f"current self-play batch: {i+1}")
                    win_ratio = 0.5
                    # win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model("./current_policy.model")
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model("./best_policy.model")
                        if (
                            self.best_win_ratio == 1.0
                            and self.pure_mcts_playout_num < 5000
                        ):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print("\n\rquit")


if __name__ == "__main__":
    training_pipeline = TrainPipeline()
    training_pipeline.run()
