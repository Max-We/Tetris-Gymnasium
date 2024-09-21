"""
@author: Junxiao Song
"""
from examples.alpha_zero.agent import MCTSAgent
from tetris_gymnasium.envs import Tetris


def start_play(env: Tetris, agent: MCTSAgent, is_shown=1):
    """start a game between two players"""
    action_space = env.action_space.n
    obs, _ = env.reset()
    if is_shown:
        env.render()
    while True:
        move = agent.get_action(obs, action_space)
        obs, reward, terminated, truncated, info = env.step(move)
        if is_shown:
            env.render()
        if terminated:
            print("Game Over")
            return


def start_self_play(env: Tetris, agent: MCTSAgent, is_shown=0, temp=1e-3):
    """start a self-play game using a MCTS player, reuse the search tree,
    and store the self-play data: (state, mcts_probs, z) for training
    """
    action_space = env.action_space.n

    obs, _ = env.reset()
    states, mcts_probs, rewards = [], [], []
    while True:
        move, move_probs = agent.get_action(env, temp=temp, return_prob=1)
        # store the data
        states.append(obs)
        mcts_probs.append(move_probs)
        # perform a move
        obs, reward, terminated, truncated, info = env.step(move)
        rewards.append(reward)
        if is_shown:
            env.render()
        if terminated:
            agent.reset_agent()
            if is_shown:
                print("Game over")
            return zip(states, mcts_probs, rewards)
