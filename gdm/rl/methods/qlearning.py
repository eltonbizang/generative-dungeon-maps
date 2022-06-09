import numpy as np
from gdm.rl.tools import Q, Policy, Trajectory, Gain


class QTable(Q):
    def __init__(self, num_states: int, num_actions: int, alpha: float):
        super().__init__(num_states, num_actions, alpha)
        self._table = np.zeros((num_states, num_actions))

    def __getitem__(self, item):
        return self._table[item]

    def __setitem__(self, key, value):
        self._table[key] = value

    def update(self, target):
        self[self.current_state, self.current_action] *= (1 - self.alpha)
        self[self.current_state, self.current_action] += (self.alpha * target)


def qlearning(q: Q, env, policy: Policy, discount_rate=0.7,
              num_episodes=10000, time_limit=np.inf,
              eval_frequency=1000, snapshot_frequency=100):
    gains = list()
    penalties = []
    evals = []
    for i in range(1, num_episodes + 1):
        q.current_state = env.reset()
        trajectory = Trajectory()
        penalty = 0
        gain = Gain(discount_rate=discount_rate)
        t = 1
        end_game = False
        win = False
        if i % eval_frequency != 0:
            # learning
            while not end_game:
                q.current_action = policy(q.current_state)
                next_state, reward, win, info = env.step(q.current_action)
                # compute target
                target = reward + discount_rate * np.max(q[next_state])
                # update the Q interface
                q.update(target=target)
                # gain update
                gain.update(reward)

                q.current_state = next_state
                t += 1
                end_game = win or (t > time_limit)

            if i % snapshot_frequency == 0:
                print(f'\rEpisode: {i}', end="")
                gains.append(gain)
                penalties.append(penalty)

            if not win:
                pass
                # penalty to apply if end_game without winning
        else:
            # evaluation
            evals.append(evaluation(policy, env))

    print("\nTraining Finished.\n")
    return q, gains, penalties, evals


def evaluation(policy, env, training=True, num_episodes=100, time_limit=100):
    """Evaluate agent's performance after Q-learning"""

    total_epochs, total_penalties = 0, 0

    for _ in range(num_episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        done = False

        while (not done) and (epochs < time_limit):
            action = policy(state)
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    avg_timesteps_per_episode = total_epochs / num_episodes
    avg_penalties_per_episode = total_penalties / num_episodes

    if not training:
        print(f"Results after the last {num_episodes} episodes:")
        print(f"\tAverage timesteps per episode: {avg_timesteps_per_episode}")
        print(f"\tAverage penalties per episode: {avg_penalties_per_episode}")

    return avg_timesteps_per_episode, avg_penalties_per_episode


