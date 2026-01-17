from snake_rl.envs.snake_grid_env import SnakeGridEnv
from snake_rl.opponent_model.hmm import HMMOpponentModel


def test_hmm_symbol_encoding():
    env = SnakeGridEnv(grid_size=7, num_snakes=2, max_steps=10, render_mode="none")
    env.food_positions = [(2, 2)]
    hmm = HMMOpponentModel()

    symbol = hmm._encode_observation_symbol(
        env=env,
        prev_opp_head=(1, 2),
        prev_opp_dir=3,
        prev_me_head=(5, 5),
        new_opp_head=(2, 2),
        new_opp_dir=3,
        new_me_head=(5, 5),
    )

    assert symbol == 13
