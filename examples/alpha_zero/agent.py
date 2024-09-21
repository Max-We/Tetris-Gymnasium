import numpy as np

from examples.alpha_zero.mcts import MCTS
from examples.alpha_zero.mcts_pure import MCTSPure, policy_value_fn
from tetris_gymnasium.envs import Tetris


class MCTSAgent:
    """AI agent based on MCTS"""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_agent(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env: Tetris, temp=1e-3, return_prob=0):
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(env.action_space.n)
        acts, probs = self.mcts.get_move_probs(env, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            # add Dirichlet Noise for exploration (needed for
            # self-play training)
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))),
            )
            # update the root node and reuse the search tree
            self.mcts.update_with_move(move)
        else:
            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob
            move = np.random.choice(acts, p=probs)
            # reset the root node
            self.mcts.update_with_move(-1)
            # location = board.move_to_location(move)
            # print("AI move: %d,%d\n" % (location[0], location[1]))

        if return_prob:
            return move, move_probs
        else:
            return move

    def __str__(self):
        return "MCTS Agent"


class MCTSAgentPure:
    """AI player based on MCTS"""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTSPure(policy_value_fn, c_puct, n_playout)

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, obs):
        move = self.mcts.get_move(obs)
        self.mcts.update_with_move(-1)
        return move

    def __str__(self):
        return "MCTS pure Agent"
