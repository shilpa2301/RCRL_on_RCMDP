class Robust_RCAC_NPG_PD:
    def __init__(self, args):
        # Initialize environment
        if args.env == 'CartPolePerturbedEnv':
            self.env = CartPolePerturbedEnv()
        elif args.env == 'CartPoleCostEnv':
            self.env = CartPoleCostEnv()
        elif args.env == 'HopperPerturbedEnv':
            self.env = HopperPerturbedEnv()
        else:
            raise ValueError("Invalid environment selected!")

        # Hyperparameters
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.gamma = args.gamma
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.lr_actor = args.lr_a
        self.lr_critic = args.lr_c
        self.lr_lambda = args.lr_lambda
        self.lambda_ = 0.0  # Dual variable (initialized to 0)
        self.baseline = args.baseline  # Cost constraint

        # Actor and Critics
        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        elif self.policy_dist == "Gaussian":
            self.actor = Actor_Gaussian(args)
        else:
            self.actor = Actor_Discrete(args)

        self.V_r = Critic(args)  # Reward critic
        self.V_c = CostCritic(args)  # Cost critic

        # Optimizers
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_reward_critic = torch.optim.Adam(self.V_r.parameters(), lr=self.lr_critic)
        self.optimizer_cost_critic = torch.optim.Adam(self.V_c.parameters(), lr=self.lr_critic)

        # State normalization
        self.state_norm = Normalization(shape=args.state_dim)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()  # Sample the action
            a_logprob = dist.log_prob(a)  # Compute log probability
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replay_buffer):
        s, a, a_logprob, r, c, s_, dw, done = replay_buffer.numpy_to_tensor()

        # ==================== Compute GAE for Reward and Cost ====================
        adv_r = []
        adv_c = []
        gae_r, gae_c = 0, 0

        with torch.no_grad():
            V_r_pred = self.V_r(s)
            V_r_next = self.V_r(s_)
            V_c_pred = self.V_c(s)
            V_c_next = self.V_c(s_)

            deltas_r = r + self.gamma * (1 - dw) * V_r_next - V_r_pred
            deltas_c = c + self.gamma * (1 - dw) * V_c_next - V_c_pred

            for delta_r, delta_c, d in zip(reversed(deltas_r.flatten().numpy()), reversed(deltas_c.flatten().numpy()), reversed(done.flatten().numpy())):
                gae_r = delta_r + self.gamma * self.lamda * gae_r * (1.0 - d)
                adv_r.insert(0, gae_r)

                gae_c = delta_c + self.gamma * self.lamda * gae_c * (1.0 - d)
                adv_c.insert(0, gae_c)

        adv_r = torch.tensor(adv_r, dtype=torch.float32).view(-1, 1)
        adv_c = torch.tensor(adv_c, dtype=torch.float32).view(-1, 1)

        # Normalize advantages
        if self.use_adv_norm:
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

        # ==================== Actor Update ====================
        dist = self.actor.get_dist(s)
        entropy = dist.entropy().sum(dim=1, keepdim=True)
        log_probs = dist.log_prob(a).sum(dim=1, keepdim=True)

        # Primal-Dual actor loss
        adv = adv_r - self.lambda_ * adv_c
        actor_loss = -(log_probs * adv).mean() - self.entropy_coef * entropy.mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:  # Trick: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer_actor.step()

        # ==================== Critic Updates ====================
        reward_critic_loss = F.mse_loss(V_r_pred, adv_r + V_r_pred)
        cost_critic_loss = F.mse_loss(V_c_pred, adv_c + V_c_pred)

        self.optimizer_reward_critic.zero_grad()
        reward_critic_loss.backward()
        self.optimizer_reward_critic.step()

        self.optimizer_cost_critic.zero_grad()
        cost_critic_loss.backward()
        self.optimizer_cost_critic.step()

        # ==================== Dual Variable Update ====================
        cost_mean = c.sum().item()  # Episodic cost
        self.lambda_ = max(0.0, self.lambda_ + self.lr_lambda * (cost_mean - self.baseline))

    def evaluate(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            if self.policy_dist == "Beta":
                a = self.actor.mean(s).detach().numpy().flatten()
            elif self.policy_dist == "Gaussian":
                a = self.actor(s).detach().numpy().flatten()
            else:
                a = self.actor(s).detach().numpy().flatten()
        return a


def train(args, run_number):
    # Directories for saving models and data
    model_dir = f"./models/{args.env}/run{run_number}/"
    data_train_dir = f"./data_train/{args.env}/run{run_number}/"
    plot_data_dir = f"./plot_data/{args.env}/run{run_number}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_train_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    # Initialize environment and normalization
    env = gym.make(args.env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    state_norm = Normalization(shape=env.observation_space.shape[0])

    # Initialize agent
    agent = Robust_RCAC_NPG_PD(args)

    episode_rewards = []
    episode_costs = []
    best_reward = float('-inf')

    for episode in range(args.max_train_steps):
        s, _ = env.reset()
        s = state_norm(s)
        done = False
        total_reward = 0
        total_cost = 0

        replay_buffer = ReplayBuffer(args)

        while not done:
            a, a_logprob = agent.choose_action(s)
            s_next, r, term, trunc, info = env.step(a)
            done = term or trunc

            c = 0.0
            if abs(s[0]) > 1:
                c = abs(s[0])
            if done:
                c += 10.0

            s_next = state_norm(s_next)
            replay_buffer.store(s, a, a_logprob, r, c, s_next, done, done)

            s = s_next
            total_reward += r
            total_cost += c

            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer)
                replay_buffer.count = 0

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)

        # Save models periodically
        if episode % 100 == 0:
            torch.save(agent.actor.state_dict(), os.path.join(model_dir, f"actor_epoch_{episode}.pth"))
            torch.save(agent.V_r.state_dict(), os.path.join(model_dir, f"value_r_epoch_{episode}.pth"))
            torch.save(agent.V_c.state_dict(), os.path.join(model_dir, f"value_c_epoch_{episode}.pth"))

        # Save training metrics periodically
        if episode % 10 == 0:
            print(f"Episode {episode} | Reward: {total_reward:.1f} | Cost: {total_cost:.1f} | Lambda: {agent.lambda_:.3f}")
            np.save(os.path.join(data_train_dir, "episode_rewards.npy"), np.array(episode_rewards))
            np.save(os.path.join(data_train_dir, "episode_costs.npy"), np.array(episode_costs))

        # Save plots periodically
        if episode % 1 == 0:
            plot_metrics(episode_rewards, episode_costs, filename=os.path.join(plot_data_dir, "training_metrics.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_train_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate for actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate for critics")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--baseline", type=float, default=9.0, help="Cost constraint threshold")
    parser.add_argument("--lr_lambda", type=float, default=1e-4, help="Learning rate for dual variable (lambda)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()

    train(args, run_number=1)
