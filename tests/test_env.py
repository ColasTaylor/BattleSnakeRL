from snake_rl.envs.snake_grid_env import SnakeGridEnv


def test_env_reset_and_step():
    env = SnakeGridEnv(grid_size=7, num_snakes=1, max_steps=10, render_mode="none")
    obs_n = env.reset()
    assert len(obs_n) == 1
    assert obs_n[0].shape == (7, 7)

    obs_n, rewards, done, info = env.step(0)
    assert len(obs_n) == 1
    assert len(rewards) == 1
    assert isinstance(done, bool)
    env.close()
