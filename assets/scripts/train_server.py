#!/usr/bin/env python3
"""
Servidor PPO con LSTM (actor) + FF (critic) para el jefe de Godot.
"""

import json
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

# ──────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────
HOST            = "127.0.0.1"
PORT            = 9999
INPUT_DIM       = 46
DISCRETE_ACTIONS = 2
HIDDEN_SIZE     = 256
SEQ_LEN         = 16

LR              = 0.00005
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.15
ADAM_EPS        = 1e-5
ENTROPY_COEF    = 0.05
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5
EPOCHS          = 6
BATCH_SIZE      = 256
BUFFER_SIZE     = 16384

REWARD_CLIP     = 10.0
OBS_CLIP        = 10.0
LOG_STD_MOVE_INIT  = -0.5
LOG_STD_ANGLE_INIT = -0.7

WIN_BUFFER_MAX    = 15    # episodios ganados a guardar
WIN_REPLAY_RATIO  = 0.3  # N% del batch de episodios ganados

MODEL_SAVE_PATH = "./assets/train_data/boss_brain.pth"
MODEL_LOAD_PATH = "./assets/train_data/boss_brain.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")


# ──────────────────────────────────────────
# Running Mean Std
# ──────────────────────────────────────────
class RunningMeanStd:
    def __init__(self, shape=()) -> None:
        self.mean  = np.zeros(shape, dtype=np.float32)
        self.var   = np.ones(shape,  dtype=np.float32)
        self.count = 1000.0

    def update(self, x: np.ndarray) -> None:
        x = np.atleast_1d(x)
        batch_mean  = np.mean(x, axis=0)
        batch_var   = np.var(x,  axis=0)
        batch_count = x.shape[0]
        delta       = batch_mean - self.mean
        tot_count   = self.count + batch_count
        self.mean   = self.mean + delta * batch_count / tot_count
        m_a         = self.var * self.count
        m_b         = batch_var * batch_count
        M2          = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var    = M2 / tot_count
        self.count  = tot_count

    def normalize(self, x: np.ndarray, clip: float = None) -> np.ndarray:
        normed = (x - self.mean) / np.sqrt(self.var + 1e-8)
        if clip is not None:
            normed = np.clip(normed, -clip, clip)
        return normed


# ──────────────────────────────────────────
# Red Actor-Crítico (Actor LSTM, Crítico FF)
# ──────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_size, discrete_actions) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # ── Actor (con memoria temporal) ──
        self.actor_fc    = nn.Linear(input_dim, hidden_size)
        self.actor_lstm  = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.move_x_mean    = nn.Linear(hidden_size, 1)
        self.move_y_mean    = nn.Linear(hidden_size, 1)
        self.angle_mean     = nn.Linear(hidden_size, 1)
        self.log_std_move   = nn.Parameter(torch.full((2,), LOG_STD_MOVE_INIT))
        self.log_std_angle  = nn.Parameter(torch.full((1,), LOG_STD_ANGLE_INIT))
        self.discrete_logits = nn.Linear(hidden_size, discrete_actions)

        # ── Crítico feedforward (más rápido, sin estado recurrente) ──
        self.critic_fc1  = nn.Linear(input_dim, hidden_size)
        self.critic_fc2  = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_head = nn.Linear(hidden_size // 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def get_initial_states(self, batch_size: int = 1) -> tuple:
        z = lambda: torch.zeros(1, batch_size, self.hidden_size, device=device)
        # El crítico FF no necesita hidden state; retornamos None para compatibilidad
        return (z(), z()), None

    def forward_actor(self, x, actor_hidden) -> tuple:
        """x: (B, T, input_dim) → salidas actor + nuevo hidden"""
        a_feat = F.relu(self.actor_fc(x))
        a_out, (nah, nac) = self.actor_lstm(a_feat, actor_hidden)
        mx       = self.move_x_mean(a_out).squeeze(-1)
        my       = self.move_y_mean(a_out).squeeze(-1)
        a_mean   = self.angle_mean(a_out).squeeze(-1)
        d_logits = self.discrete_logits(a_out)
        return mx, my, a_mean, d_logits, (nah, nac)

    def forward_critic(self, x) -> torch.Tensor:
        """
        x: (B, T, input_dim) o (B, input_dim)
        Retorna value (B, T) o (B,) según input.
        El crítico opera sobre cada paso independientemente (sin memoria).
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
            c = F.relu(self.critic_fc1(x_flat))
            c = F.relu(self.critic_fc2(c))
            v = self.critic_head(c).squeeze(-1)
            return v.reshape(B, T)
        else:
            c = F.relu(self.critic_fc1(x))
            c = F.relu(self.critic_fc2(c))
            return self.critic_head(c).squeeze(-1)

    def get_action_and_logprob(self, state, actor_hidden, _critic_hidden=None,
                                deterministic=False) -> tuple:
        x = state.unsqueeze(1)  # (1, 1, input_dim)
        mx, my, a_mean, d_logits, new_ah = self.forward_actor(x, actor_hidden)

        mx, my, a_mean = mx[:, -1], my[:, -1], a_mean[:, -1]
        d_logits = d_logits[:, -1, :]
        val = self.forward_critic(state)  # (1,) — sin secuencia

        move_std  = torch.exp(self.log_std_move)
        angle_std = torch.exp(self.log_std_angle)
        move_dist  = Normal(torch.stack([mx, my], dim=-1), move_std)
        angle_dist = Normal(a_mean, angle_std)
        disc_dist  = Categorical(logits=d_logits)

        if deterministic:
            move_xy    = torch.stack([mx, my], dim=-1)
            angle      = a_mean
            action_idx = torch.argmax(d_logits, dim=-1)
            log_prob   = None
        else:
            move_xy    = move_dist.rsample()
            angle      = angle_dist.rsample()
            action_idx = disc_dist.sample()
            log_prob   = (move_dist.log_prob(move_xy).sum(-1)
                          + angle_dist.log_prob(angle)
                          + disc_dist.log_prob(action_idx))

        move_xy = torch.tanh(move_xy)
        angle   = torch.tanh(angle)

        action_dict = {
            'move_x':     move_xy[0, 0].item(),
            'move_y':     move_xy[0, 1].item(),
            'shot_angle': angle[0].item(),
            'action':     action_idx[0].item(),
        }
        return action_dict, log_prob, val, new_ah, None

    def evaluate(self, states, actions, actor_hidden, _critic_hidden=None) -> tuple:
        """states: (B, T, input_dim)"""
        mx, my, a_mean, d_logits, _ = self.forward_actor(states, actor_hidden)
        vals = self.forward_critic(states)  # (B, T)

        B, T = mx.shape
        mx       = mx.reshape(B * T)
        my       = my.reshape(B * T)
        a_mean   = a_mean.reshape(B * T)
        d_logits = d_logits.reshape(B * T, -1)
        vals     = vals.reshape(B * T)

        move_std   = torch.exp(self.log_std_move)
        angle_std  = torch.exp(self.log_std_angle)
        move_dist  = Normal(torch.stack([mx, my], dim=-1), move_std)
        angle_dist = Normal(a_mean, angle_std)
        disc_dist  = Categorical(logits=d_logits)

        move_xy    = torch.stack([actions['move_x'].reshape(B*T),
                                   actions['move_y'].reshape(B*T)], dim=-1)
        shot_angle = actions['shot_angle'].reshape(B*T)
        action_idx = actions['action'].reshape(B*T)

        log_probs = (move_dist.log_prob(move_xy).sum(-1)
                     + angle_dist.log_prob(shot_angle)
                     + disc_dist.log_prob(action_idx))

        entropy = (move_dist.entropy().sum(-1).mean()
                   + angle_dist.entropy().mean()
                   + disc_dist.entropy().mean())

        return log_probs, vals, entropy


# ──────────────────────────────────────────
# Buffer de experiencia
# ──────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, buffer_size, gamma, gae_lambda, seq_len) -> None:
        self.buffer_size = buffer_size
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.seq_len     = seq_len
        self.clear()

    def clear(self) -> None:
        self.states        = []
        self.actions       = []
        self.rewards       = []
        self.dones         = []
        self.log_probs     = []
        self.values        = []
        self.actor_hidden  = []

    def add(self, state, action, reward, done, log_prob, value, ah, _ch=None) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.actor_hidden.append((ah[0].detach().clone(), ah[1].detach().clone()))

    def ready(self) -> bool:
        return len(self.states) >= self.buffer_size

    def compute_advantages_and_returns(self, last_value: float = 0.0) -> tuple:
        rewards = np.array(self.rewards, dtype=np.float32)
        dones   = np.array(self.dones,   dtype=np.float32)
        values  = np.array(self.values + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta         = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae           = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns    = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = np.clip(advantages, -5.0, 5.0)
        return advantages, returns

    def get_training_batches(self, advantages, returns, batch_size,
                              extra_states=None, extra_actions=None,
                              extra_log_probs=None, extra_advantages=None,
                              extra_returns=None, extra_hidden=None) -> any:
        n = len(self.states)
        T = self.seq_len
        starts = list(range(0, n - T + 1, T))
        np.random.shuffle(starts)

        # Si hay experiencia de victorias, mezclamos proporcionalmente
        win_starts = []
        if extra_states and len(extra_states) >= T:
            n_win    = len(extra_states)
            win_pool = list(range(0, n_win - T + 1, T))
            np.random.shuffle(win_pool)
            # Cuántos chunks de victoria por batch
            chunks_per_batch = max(1, batch_size // T)
            win_per_batch    = max(1, int(chunks_per_batch * WIN_REPLAY_RATIO))
            win_starts       = win_pool[:win_per_batch * (len(starts) // chunks_per_batch + 1)]

        chunks_per_batch = max(1, batch_size // T)

        for i in range(0, len(starts), chunks_per_batch):
            batch_starts = starts[i: i + chunks_per_batch]
            if not batch_starts:
                continue

            s_list  = []
            a_dict  = {k: [] for k in ('move_x','move_y','shot_angle','action')}
            lp_list = []
            adv_list = []
            ret_list = []
            ah_list  = []

            # Chunks del buffer principal
            for s in batch_starts:
                idx = list(range(s, s + T))
                s_list.append([self.states[j] for j in idx])
                for k in a_dict:
                    a_dict[k].append([self.actions[j][k] for j in idx])
                lp_list.append([self.log_probs[j]  for j in idx])
                adv_list.append([advantages[j]      for j in idx])
                ret_list.append([returns[j]          for j in idx])
                ah_list.append(self.actor_hidden[s])

            # Chunks de victorias (si hay)
            if extra_states and win_starts:
                win_chunk_idx = (i // chunks_per_batch) % max(1, len(win_starts) // max(1, int(chunks_per_batch * WIN_REPLAY_RATIO)))
                for wc in range(int(chunks_per_batch * WIN_REPLAY_RATIO)):
                    ws_idx = (win_chunk_idx * int(chunks_per_batch * WIN_REPLAY_RATIO) + wc) % max(1, len(win_starts))
                    ws = win_starts[ws_idx]
                    if ws + T > len(extra_states):
                        continue
                    idx = list(range(ws, ws + T))
                    s_list.append([extra_states[j]  for j in idx])
                    for k in a_dict:
                        a_dict[k].append([extra_actions[j][k] for j in idx])
                    lp_list.append([extra_log_probs[j]   for j in idx])
                    adv_list.append([extra_advantages[j]  for j in idx])
                    ret_list.append([extra_returns[j]      for j in idx])
                    ah_list.append(extra_hidden[ws])

            states_t = torch.FloatTensor(np.array(s_list)).to(device)
            lp_old_t = torch.FloatTensor(np.array(lp_list)).to(device)
            adv_t    = torch.FloatTensor(np.array(adv_list)).to(device)
            ret_t    = torch.FloatTensor(np.array(ret_list)).to(device)

            actions_t = {
                'move_x':     torch.FloatTensor(np.array(a_dict['move_x'])).to(device),
                'move_y':     torch.FloatTensor(np.array(a_dict['move_y'])).to(device),
                'shot_angle': torch.FloatTensor(np.array(a_dict['shot_angle'])).to(device),
                'action':     torch.LongTensor(np.array(a_dict['action'])).to(device),
            }

            ah0 = torch.cat([h[0] for h in ah_list], dim=1)
            ac0 = torch.cat([h[1] for h in ah_list], dim=1)

            yield states_t, actions_t, lp_old_t, adv_t, ret_t, (ah0, ac0), None


# ──────────────────────────────────────────
# Win Episode Buffer
# ──────────────────────────────────────────
class WinBuffer:
    """Guarda los últimos N episodios ganados completos para replay."""
    def __init__(self, max_episodes: int) -> None:
        self.max_episodes = max_episodes
        self.episodes = []  # lista de dicts con estados/acciones/etc del episodio

    def add_episode(self, states, actions, log_probs, values, rewards,
                    dones, actor_hiddens, gamma, gae_lambda) -> None:
        # Calcular ventajas y returns para este episodio
        r = np.array(rewards, dtype=np.float32)
        d = np.array(dones,   dtype=np.float32)
        v = np.array(values + [0.0], dtype=np.float32)

        advantages = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            delta         = r[t] + gamma * v[t+1] * (1 - d[t]) - v[t]
            gae           = delta + gamma * gae_lambda * (1 - d[t]) * gae
            advantages[t] = gae

        returns    = advantages + v[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = np.clip(advantages, -5.0, 5.0)

        self.episodes.append({
            'states':        states,
            'actions':       actions,
            'log_probs':     log_probs,
            'advantages':    advantages.tolist(),
            'returns':       returns.tolist(),
            'actor_hiddens': actor_hiddens,
        })
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def get_flat(self) -> tuple:
        """Devuelve todos los pasos de todos los episodios ganados aplanados."""
        if not self.episodes:
            return None, None, None, None, None, None
        all_s, all_a, all_lp, all_adv, all_ret, all_ah = [], [], [], [], [], []
        for ep in self.episodes:
            all_s.extend(ep['states'])
            all_a.extend(ep['actions'])
            all_lp.extend(ep['log_probs'])
            all_adv.extend(ep['advantages'])
            all_ret.extend(ep['returns'])
            all_ah.extend(ep['actor_hiddens'])
        return all_s, all_a, all_lp, all_adv, all_ret, all_ah


# ──────────────────────────────────────────
# Agente PPO
# ──────────────────────────────────────────
class PPOAgent:
    def __init__(self, config) -> None:
        self.config  = config
        self.network = ActorCritic(
            config['input_dim'], config['hidden_size'], config['discrete_actions']
        ).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config['lr'],
                                    eps=config.get('adam_eps', 1e-5))
        self.buffer     = RolloutBuffer(config['buffer_size'], config['gamma'],
                                        config['gae_lambda'], config['seq_len'])
        self.win_buffer = WinBuffer(WIN_BUFFER_MAX)

        self.obs_rms    = RunningMeanStd(shape=(INPUT_DIM,))
        self.reward_rms = RunningMeanStd(shape=())

        self.episode_count  = 0
        self.total_steps    = 0
        self.recent_rewards = []

        self.actor_h, _ = self.network.get_initial_states()

        if config.get('model_load_path'):
            try:
                self.load_model(config['model_load_path'])
                print(f"Modelo cargado desde {config['model_load_path']}")
            except FileNotFoundError:
                print("No se encontró modelo previo, empezando desde cero.")

    def reset_hidden_states(self) -> None:
        self.actor_h, _ = self.network.get_initial_states()

    def _normalize_obs(self, obs: list) -> np.ndarray:
        return self.obs_rms.normalize(np.array(obs, dtype=np.float32), clip=OBS_CLIP)

    def _normalize_reward(self, reward: float) -> float:
        self.reward_rms.update(np.array([reward], dtype=np.float32))
        return float(np.clip(reward / (np.sqrt(self.reward_rms.var + 1e-8)),
                              -REWARD_CLIP, REWARD_CLIP))

    def get_action(self, state: list, deterministic: bool = False) -> tuple:
        norm = self._normalize_obs(state)
        s_t  = torch.FloatTensor(norm).unsqueeze(0).to(device)

        with torch.no_grad():
            action_dict, log_prob, value, new_ah, _ = \
                self.network.get_action_and_logprob(s_t, self.actor_h, None, deterministic)

        log_prob_val = log_prob.item() if log_prob is not None else 0.0
        value_val    = value.item()
        old_ah       = (self.actor_h[0].clone(), self.actor_h[1].clone())
        self.actor_h = new_ah

        return action_dict, log_prob_val, value_val, old_ah, None

    def store_transition(self, state, action, reward, done, log_prob, value, ah, _ch=None) -> None:
        norm_state  = self._normalize_obs(state)
        norm_reward = self._normalize_reward(reward)
        self.obs_rms.update(np.array([state], dtype=np.float32))
        self.buffer.add(norm_state, action, norm_reward, done, log_prob, value, ah)
        self.total_steps += 1

    def update(self, last_value: float = 0.0, boss_won: bool = False,
               episode_data: dict = None) -> None:
        if len(self.buffer.states) < self.config['seq_len']:
            return

        # Si el boss ganó, guardamos este episodio en el win buffer
        if boss_won and episode_data:
            self.win_buffer.add_episode(
                episode_data['states'], episode_data['actions'],
                episode_data['log_probs'], episode_data['values'],
                episode_data['rewards'], episode_data['dones'],
                episode_data['actor_hiddens'],
                self.config['gamma'], self.config['gae_lambda']
            )
            print(f"  → Victoria guardada en win_buffer ({len(self.win_buffer.episodes)}/{WIN_BUFFER_MAX})")

        advantages, returns = self.buffer.compute_advantages_and_returns(last_value)

        # Obtener datos de victorias para replay
        ws, wa, wlp, wadv, wret, wah = self.win_buffer.get_flat()

        for _ in range(self.config['epochs']):
            for (states_b, actions_b, lp_old_b, adv_b, ret_b, ah_b, _) in \
                    self.buffer.get_training_batches(
                        advantages, returns, self.config['batch_size'],
                        extra_states=ws, extra_actions=wa, extra_log_probs=wlp,
                        extra_advantages=wadv, extra_returns=wret, extra_hidden=wah
                    ):

                B, T = states_b.shape[:2]
                log_probs_new, values_new, entropy = self.network.evaluate(
                    states_b, actions_b, ah_b, None
                )

                lp_old_flat = lp_old_b.reshape(B * T)
                adv_flat    = adv_b.reshape(B * T)
                ret_flat    = ret_b.reshape(B * T)

                ratio  = torch.exp(log_probs_new - lp_old_flat)
                surr1  = ratio * adv_flat
                surr2  = torch.clamp(ratio, 1 - self.config['clip_eps'],
                                            1 + self.config['clip_eps']) * adv_flat

                actor_loss   = -torch.min(surr1, surr2).mean()
                value_loss   = F.mse_loss(values_new, ret_flat)
                entropy_loss = -self.config['entropy_coef'] * entropy
                loss         = actor_loss + self.config['value_coef'] * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()

        self.buffer.clear()

    def end_episode(self, total_ep_reward: float, last_state=None,
                    timed_out: bool = False, boss_won: bool = False,
                    episode_data: dict = None) -> None:
        last_value = 0.0
        if timed_out and last_state is not None:
            norm  = self._normalize_obs(last_state)
            s_t   = torch.FloatTensor(norm).unsqueeze(0).to(device)
            s_seq = s_t.unsqueeze(1)
            with torch.no_grad():
                mx, my, a_mean, d_logits, new_ah = self.network.forward_actor(s_seq, self.actor_h)
                # El crítico no es recurrente, usamos s_t directamente
                last_value = self.network.forward_critic(s_t).item()

        self.update(last_value, boss_won=boss_won, episode_data=episode_data)
        self.reset_hidden_states()

        self.recent_rewards.append(total_ep_reward)
        if len(self.recent_rewards) > 20:
            self.recent_rewards.pop(0)
        avg = sum(self.recent_rewards) / len(self.recent_rewards)
        win_str = "BOSS GANÓ" if boss_won else ""
        print(f"Ep {self.episode_count:4d} | avg20: {avg:8.1f} | "
              f"steps: {self.total_steps:7d} | reward: {total_ep_reward:8.1f}{win_str}")

        if self.episode_count % 10 == 0:
            self.save_model(self.config['model_save_path'])

        self.episode_count += 1

    def save_model(self, path) -> None:
        torch.save({
            'network_state_dict':   self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count':        self.episode_count,
            'total_steps':          self.total_steps,
            'obs_rms_mean':         self.obs_rms.mean.tolist(),
            'obs_rms_var':          self.obs_rms.var.tolist(),
            'obs_rms_count':        float(self.obs_rms.count),
            'reward_rms_mean':      self.reward_rms.mean.tolist(),
            'reward_rms_var':       self.reward_rms.var.tolist(),
            'reward_rms_count':     float(self.reward_rms.count),
        }, path)
        print(f"Modelo guardado en {path}")

    def load_model(self, path) -> None:
        chk = torch.load(path, map_location=device, weights_only=False)
        try:
            self.network.load_state_dict(chk['network_state_dict'])
        except RuntimeError:
            print("Arquitectura cambiada. Empezando desde cero.")
            return
        self.optimizer.load_state_dict(chk['optimizer_state_dict'])
        self.episode_count = chk.get('episode_count', 0)
        self.total_steps   = chk.get('total_steps',   0)
        if 'obs_rms_mean' in chk:
            self.obs_rms.mean  = np.array(chk['obs_rms_mean'], dtype=np.float32)
            self.obs_rms.var   = np.array(chk['obs_rms_var'],  dtype=np.float32)
            self.obs_rms.count = chk['obs_rms_count']
        if 'reward_rms_mean' in chk:
            self.reward_rms.mean  = np.array(chk['reward_rms_mean'], dtype=np.float32)
            self.reward_rms.var   = np.array(chk['reward_rms_var'],  dtype=np.float32)
            self.reward_rms.count = chk['reward_rms_count']


# ──────────────────────────────────────────
# Servidor UDP
# ──────────────────────────────────────────
class GodotRLServer:
    def __init__(self, agent, host, port) -> None:
        self.agent   = agent
        self.sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.1)
        self.running = True
        self._reset_step_state()

    def _reset_step_state(self) -> None:
        self.current_state    = None
        self.current_action   = None
        self.current_log_prob = None
        self.current_value    = None
        self.current_ah       = None
        self.episode_reward   = 0.0
        # Acumuladores para win buffer
        self._ep_states        = []
        self._ep_actions       = []
        self._ep_log_probs     = []
        self._ep_values        = []
        self._ep_rewards       = []
        self._ep_dones         = []
        self._ep_actor_hiddens = []

    def run(self) -> None:
        print(f"Servidor escuchando en {HOST}:{PORT}")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65536)
                self.handle_message(data, addr)
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break

    def handle_message(self, data, addr) -> None:
        try:
            msg = json.loads(data.decode('utf-8'))
        except Exception:
            return
        t = msg.get('type')
        if t == 'step':
            self.handle_step(msg, addr)
        elif t == 'episode_end':
            self.handle_episode_end(msg, addr)

    def handle_step(self, msg, addr) -> None:
        inputs = msg.get('inputs', [0.0] * INPUT_DIM)
        reward = msg.get('reward', 0.0)
        self.episode_reward += reward

        if self.current_state is not None:
            norm_state  = self.agent._normalize_obs(self.current_state)
            norm_reward = self.agent._normalize_reward(reward)
            self.agent.obs_rms.update(np.array([self.current_state], dtype=np.float32))
            self.agent.buffer.add(norm_state, self.current_action, norm_reward,
                                  False, self.current_log_prob, self.current_value,
                                  self.current_ah)
            self.agent.total_steps += 1

            # Acumular para win buffer
            self._ep_states.append(norm_state)
            self._ep_actions.append(self.current_action)
            self._ep_log_probs.append(self.current_log_prob)
            self._ep_values.append(self.current_value)
            self._ep_rewards.append(norm_reward)
            self._ep_dones.append(0.0)
            self._ep_actor_hiddens.append(
                (self.current_ah[0].clone(), self.current_ah[1].clone())
            )

        action, log_p, val, ah, _ = self.agent.get_action(inputs)
        self.current_state    = inputs
        self.current_action   = action
        self.current_log_prob = log_p
        self.current_value    = val
        self.current_ah       = ah

        self.sock.sendto(json.dumps({
            'move_x':     action['move_x'],
            'move_y':     action['move_y'],
            'shot_angle': action['shot_angle'],
            'action':     action['action'],
            'move_dir':   [action['move_x'], action['move_y']],
        }).encode('utf-8'), addr)

    def handle_episode_end(self, msg, addr) -> None:
        final_reward = msg.get('reward',    0.0)
        timed_out    = msg.get('timed_out', False)
        boss_won     = msg.get('boss_won',  False)
        self.episode_reward += final_reward
        last_state = self.current_state

        episode_data = None
        if self.current_state is not None:
            norm_state  = self.agent._normalize_obs(self.current_state)
            norm_reward = self.agent._normalize_reward(final_reward)
            self.agent.obs_rms.update(np.array([self.current_state], dtype=np.float32))
            self.agent.buffer.add(norm_state, self.current_action, norm_reward,
                                  True, self.current_log_prob, self.current_value,
                                  self.current_ah)
            self.agent.total_steps += 1

            self._ep_states.append(norm_state)
            self._ep_actions.append(self.current_action)
            self._ep_log_probs.append(self.current_log_prob)
            self._ep_values.append(self.current_value)
            self._ep_rewards.append(norm_reward)
            self._ep_dones.append(1.0)
            self._ep_actor_hiddens.append(
                (self.current_ah[0].clone(), self.current_ah[1].clone())
            )

            if boss_won:
                episode_data = {
                    'states':        self._ep_states,
                    'actions':       self._ep_actions,
                    'log_probs':     self._ep_log_probs,
                    'values':        self._ep_values,
                    'rewards':       self._ep_rewards,
                    'dones':         self._ep_dones,
                    'actor_hiddens': self._ep_actor_hiddens,
                }

        self.agent.end_episode(self.episode_reward, last_state, timed_out,
                               boss_won=boss_won, episode_data=episode_data)
        self._reset_step_state()
        self.sock.sendto(json.dumps({'type': 'episode_ready'}).encode('utf-8'), addr)

    def stop(self) -> None:
        self.running = False
        self.sock.close()


# ──────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────
if __name__ == "__main__":
    config = {
        'input_dim':        INPUT_DIM,
        'discrete_actions': DISCRETE_ACTIONS,
        'hidden_size':      HIDDEN_SIZE,
        'seq_len':          SEQ_LEN,
        'lr':               LR,
        'adam_eps':         ADAM_EPS,
        'gamma':            GAMMA,
        'gae_lambda':       GAE_LAMBDA,
        'clip_eps':         CLIP_EPS,
        'entropy_coef':     ENTROPY_COEF,
        'value_coef':       VALUE_COEF,
        'max_grad_norm':    MAX_GRAD_NORM,
        'epochs':           EPOCHS,
        'batch_size':       BATCH_SIZE,
        'buffer_size':      BUFFER_SIZE,
        'model_save_path':  MODEL_SAVE_PATH,
        'model_load_path':  MODEL_LOAD_PATH,
    }
    agent  = PPOAgent(config)
    server = GodotRLServer(agent, HOST, PORT)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nApagando servidor...")
        server.stop()
        agent.save_model(config['model_save_path'])
        print("Servidor detenido.")
    except Exception as e:
        print(f"ERROR en servidor: {e}")
        import traceback
        traceback.print_exc()