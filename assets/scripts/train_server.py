#!/usr/bin/env python3
"""
Servidor de aprendizaje por refuerzo (PPO) para el jefe de Godot.
Salidas: move_x, move_y (continuos en [-1,1]), shot_angle (en radianes, [-π,π]), action (discreto: 0=nada, 1=disparar)
"""

import json
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

# ------------------------------
# Configuración (ajústala aquí)
# ------------------------------
HOST = "127.0.0.1"
PORT = 9999
INPUT_DIM = 37                # Coincide con GlobalConst.INPUTS
DISCRETE_ACTIONS = 2          # 0: nada, 1: ataque
HIDDEN_SIZE = 256   
LR = 0.00005
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.15
ENTROPY_COEF = 0.03
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
EPOCHS = 10
BATCH_SIZE = 256
BUFFER_SIZE = 8192
MODEL_SAVE_PATH = "./assets/train_data/boss_brain.pth"
MODEL_LOAD_PATH = "./assets/train_data/boss_brain.pth"   # None para empezar de cero

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RunningMeanStd:
    def __init__(self, shape=()) -> None:
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1000.0

    def update(self, x) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.mean, self.var, self.count = self._update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @staticmethod
    def _update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count) -> tuple:
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

# ------------------------------
# Red neuronal Actor-Crítico
# ------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_size, discrete_actions) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        # --- ACTOR ---
        self.actor_fc = nn.Linear(input_dim, hidden_size)
        self.actor_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.move_x_mean = nn.Linear(hidden_size, 1)
        self.move_y_mean = nn.Linear(hidden_size, 1)
        self.angle_mean = nn.Linear(hidden_size, 1)
        self.log_std_move = nn.Parameter(torch.full((2,), -0.5))
        self.log_std_angle = nn.Parameter(torch.full((1,), -0.7))
        self.discrete_logits = nn.Linear(hidden_size, discrete_actions)
        
        # --- CRÍTICO ---
        self.critic_fc = nn.Linear(input_dim, hidden_size)
        self.critic_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.critic_head = nn.Linear(hidden_size, 1)
    
    def get_initial_states(self, batch_size=1) -> tuple:
        """Retorna los estados iniciales h0, c0 para el Actor y el Crítico"""
        ah0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        ac0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        ch0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        cc0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (ah0, ac0), (ch0, cc0)

    def forward(self, x, actor_hidden, critic_hidden) -> tuple:
        # Flujo del Actor
        a_feat = F.relu(self.actor_fc(x))
        a_out, (nah, nac) = self.actor_lstm(a_feat, actor_hidden)
        a_out_sq = a_out.squeeze(1) # Eliminamos seq_len (que será 1 en este enfoque)
        
        mx = self.move_x_mean(a_out_sq).squeeze(-1)
        my = self.move_y_mean(a_out_sq).squeeze(-1)
        a_mean = self.angle_mean(a_out_sq).squeeze(-1)
        d_logits = self.discrete_logits(a_out_sq)
        
        # Flujo del Crítico
        c_feat = F.relu(self.critic_fc(x))
        c_out, (nch, ncc) = self.critic_lstm(c_feat, critic_hidden)
        c_out_sq = c_out.squeeze(1)
        
        value = self.critic_head(c_out_sq).squeeze(-1)

        return mx, my, a_mean, d_logits, value, (nah, nac), (nch, ncc)

    def get_action_and_logprob(self, state, actor_hidden, critic_hidden, deterministic=False) -> tuple:
        # Añadir dimensión de secuencia: (1, input_dim) -> (1, 1, input_dim)
        state_seq = state.unsqueeze(1)
        
        mx, my, a_mean, d_logits, val, new_ah, new_ch = self.forward(state_seq, actor_hidden, critic_hidden)
        
        move_std = torch.exp(self.log_std_move)
        angle_std = torch.exp(self.log_std_angle)

        move_dist = Normal(torch.stack([mx, my], dim=-1), move_std)
        angle_dist = Normal(a_mean, angle_std)
        discrete_dist = Categorical(logits=d_logits)

        if deterministic:
            move_xy = torch.stack([mx, my], dim=-1)
            angle = a_mean
            action_idx = torch.argmax(d_logits, dim=-1)
            log_prob = None
        else:
            move_xy = move_dist.rsample()          
            angle = angle_dist.rsample()           
            action_idx = discrete_dist.sample()    
            log_prob_move = move_dist.log_prob(move_xy).sum(dim=-1)
            log_prob_angle = angle_dist.log_prob(angle)
            log_prob_discrete = discrete_dist.log_prob(action_idx)
            log_prob = log_prob_move + log_prob_angle + log_prob_discrete

        move_xy = torch.tanh(move_xy) 
        angle = torch.tanh(angle)     
        
        action_dict = {
            'move_x': move_xy[0, 0].item(),
            'move_y': move_xy[0, 1].item(),
            'shot_angle': angle[0].item(),
            'action': action_idx[0].item()
        }
        return action_dict, log_prob, val, new_ah, new_ch

    def evaluate(self, states, actions, actor_hiddens, critic_hiddens) -> tuple:
        # states shape: (batch, 1, input_dim)
        mx, my, a_mean, d_logits, vals, _, _ = self.forward(states, actor_hiddens, critic_hiddens)
        
        move_std = torch.exp(self.log_std_move)
        angle_std = torch.exp(self.log_std_angle)
        move_xy = torch.stack([actions['move_x'], actions['move_y']], dim=-1)
        move_dist = Normal(torch.stack([mx, my], dim=-1), move_std)
        angle_dist = Normal(a_mean, angle_std)
        discrete_dist = Categorical(logits=d_logits)

        log_prob_move = move_dist.log_prob(move_xy).sum(dim=-1)
        log_prob_angle = angle_dist.log_prob(actions['shot_angle'])
        log_prob_discrete = discrete_dist.log_prob(actions['action'])
        log_probs = log_prob_move + log_prob_angle + log_prob_discrete

        entropy = (move_dist.entropy().sum(dim=-1).mean() +
                   angle_dist.entropy().mean() +
                   discrete_dist.entropy().mean())
        return log_probs, vals, entropy

# ------------------------------
# Buffer de experiencia
# ------------------------------
class RolloutBuffer:
    def __init__(self, buffer_size, gamma, gae_lambda) -> None:
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        self.actor_hidden = []
        self.critic_hidden = []

    def add(self, state, action, reward, done, log_prob, value, ah, ch) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        self.actor_hidden.append((ah[0].detach(), ah[1].detach()))
        self.critic_hidden.append((ch[0].detach(), ch[1].detach()))

    def ready(self) -> bool:
        return len(self.states) >= self.buffer_size

    def compute_advantages_and_returns(self, last_value=0.0)  -> tuple:
        advantages = []
        returns = []
        gae = 0.0
        rewards = np.array(self.rewards)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [last_value])

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            ret = rewards[t] + self.gamma * (returns[0] if returns else values[t+1]) * (1 - dones[t])
            returns.insert(0, ret)

        advantages = np.array(advantages)
        returns = np.array(returns)
        # Normalizar ventajas
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def get_training_batches(self, advantages, returns, batch_size) -> any:
        # Mezcla aleatoria de pasos, permitida porque proporcionamos el hidden state exacto 
        # en el que se encontraba el agente en cada paso (Truncated BPTT con seq_len=1)
        indices = np.random.permutation(len(self.states))
        for start in range(0, len(self.states), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # Reconstruir batch añadiendo la dimensión de secuencia (batch, 1, input)
            states_batch = torch.FloatTensor(np.array(self.states)[batch_idx]).unsqueeze(1).to(device)
            actions_batch = {
                'move_x': torch.FloatTensor([self.actions[i]['move_x'] for i in batch_idx]).to(device),
                'move_y': torch.FloatTensor([self.actions[i]['move_y'] for i in batch_idx]).to(device),
                'shot_angle': torch.FloatTensor([self.actions[i]['shot_angle'] for i in batch_idx]).to(device),
                'action': torch.LongTensor([self.actions[i]['action'] for i in batch_idx]).to(device)
            }
            log_probs_old = torch.FloatTensor(np.array(self.log_probs)[batch_idx]).to(device)
            adv_batch = torch.FloatTensor(advantages[batch_idx]).to(device)
            ret_batch = torch.FloatTensor(returns[batch_idx]).to(device)
            
            ah0_batch = torch.cat([self.actor_hidden[i][0] for i in batch_idx], dim=1)
            ac0_batch = torch.cat([self.actor_hidden[i][1] for i in batch_idx], dim=1)
            ch0_batch = torch.cat([self.critic_hidden[i][0] for i in batch_idx], dim=1)
            cc0_batch = torch.cat([self.critic_hidden[i][1] for i in batch_idx], dim=1)

            yield states_batch, actions_batch, log_probs_old, adv_batch, ret_batch, (ah0_batch, ac0_batch), (ch0_batch, cc0_batch)

# ------------------------------
# Agente PPO
# ------------------------------
class PPOAgent:
    def __init__(self, config) -> None:
        self.config = config
        self.network = ActorCritic(config['input_dim'], config['hidden_size'], config['discrete_actions']).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config['lr'])
        self.buffer = RolloutBuffer(config['buffer_size'], config['gamma'], config['gae_lambda'])
        self.episode_count = 0
        self.total_steps = 0
        self.recent_rewards = []
        self.obs_rms = RunningMeanStd(shape=(INPUT_DIM,))
        self.reward_scale = 0.05   # divide la recompensa entre 10
        
        self.actor_h, self.critic_h = self.network.get_initial_states()
        
        if config['model_load_path']:
            try:
                self.load_model(config['model_load_path'])
                print(f"Modelo cargado desde {config['model_load_path']}")
            except FileNotFoundError:
                print("No se encontró modelo previo, empezando desde cero.")
    
    def reset_hidden_states(self) -> None:
        """Limpia la memoria del LSTM al inicio de un nuevo episodio."""
        self.actor_h, self.critic_h = self.network.get_initial_states()
    
    def get_action(self, state, deterministic=False)  -> tuple:
        norm_state = (state - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        state_t = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Pasamos los hidden states actuales y capturamos los nuevos
            action_dict, log_prob, value, new_ah, new_ch = self.network.get_action_and_logprob(
                state_t, self.actor_h, self.critic_h, deterministic
            )
            
        log_prob_val = log_prob.item() if log_prob is not None else 0.0
        value_val = value.item()
        
        # Guarda copias desconectadas de los states actuales antes de actualizarlos.
        old_actor_h = (self.actor_h[0].clone(), self.actor_h[1].clone())
        old_critic_h = (self.critic_h[0].clone(), self.critic_h[1].clone())
        
        # Actualiza la memoria del agente con lo que "pensó" en este paso
        self.actor_h = new_ah
        self.critic_h = new_ch

        return action_dict, log_prob_val, value_val, old_actor_h, old_critic_h

    def store_transition(self, state, action, reward, done, log_prob, value, ah, ch) -> None:
        norm_state = (state - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        scaled_reward = reward * self.reward_scale
        self.buffer.add(norm_state, action, scaled_reward, done, log_prob, value, ah, ch)
        self.total_steps += 1
        self.obs_rms.update(np.array([state]))

    def update(self, last_value=0.0) -> None:
        if len(self.buffer.states) == 0:
            return
        advantages, returns = self.buffer.compute_advantages_and_returns(last_value)
        for _ in range(self.config['epochs']):
            for (states_batch, actions_batch, log_probs_old, adv_batch, ret_batch, ah_batch, ch_batch) in \
                    self.buffer.get_training_batches(advantages, returns, self.config['batch_size']):
                
                log_probs_new, values_new, entropy = self.network.evaluate(
                    states_batch, actions_batch, ah_batch, ch_batch
                )
                
                ratio = torch.exp(log_probs_new - log_probs_old)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.config['clip_eps'], 1 + self.config['clip_eps']) * adv_batch
                
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_new, ret_batch)
                entropy_loss = -self.config['entropy_coef'] * entropy
                loss = actor_loss + self.config['value_coef'] * value_loss + entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
        self.buffer.clear()

    def end_episode(self, total_ep_reward:float, last_state=None, timed_out: bool = False) -> None:
        last_value = 0.0
        if timed_out and last_state is not None:
            norm_state = (np.array(last_state) - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            state_t = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
            state_seq = state_t.unsqueeze(1)
            with torch.no_grad():
                _, _, _, _, value, _, _ = self.network.forward(state_seq, self.actor_h, self.critic_h)
                last_value = value.item()
                
        self.update(last_value)
        
        # Reiniciar memoria a corto plazo para el nuevo intento
        self.reset_hidden_states()
        
        self.recent_rewards.append(total_ep_reward)
        if len(self.recent_rewards) > 20:
            self.recent_rewards.pop(0)
        avg = sum(self.recent_rewards) / len(self.recent_rewards)
        print(f"Ep {self.episode_count} | avg(20): {avg:.1f} | steps: {self.total_steps} | total_reward: {total_ep_reward}")
        
        if self.episode_count % 10 == 0:
            self.save_model(self.config['model_save_path'])

        self.episode_count += 1

    def save_model(self, path) -> None:
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'obs_rms_mean': self.obs_rms.mean.tolist(),
            'obs_rms_var': self.obs_rms.var.tolist(),
            'obs_rms_count': float(self.obs_rms.count)
        }, path)
        print(f"Modelo guardado en {path}")

    def load_model(self, path) -> None:
        chk = torch.load(path, map_location=device, weights_only=False)
        try:
            self.network.load_state_dict(chk['network_state_dict'])
        except RuntimeError:
            print("Arquitectura cambiada detectada. Se empezará desde cero para evitar conflictos de pesos.")
            return
            
        self.optimizer.load_state_dict(chk['optimizer_state_dict'])
        self.episode_count = chk.get('episode_count', 0)
        self.total_steps = chk.get('total_steps', 0)
        if 'obs_rms_mean' in chk:
            self.obs_rms.mean  = np.array(chk['obs_rms_mean'], dtype=np.float32)
            self.obs_rms.var   = np.array(chk['obs_rms_var'],  dtype=np.float32)
            self.obs_rms.count = chk['obs_rms_count']

# ------------------------------
# Servidor UDP
# ------------------------------
class GodotRLServer:
    def __init__(self, agent, host, port) -> None:
        self.agent = agent
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.1)
        self.running = True
        
        self.current_state = None
        self.current_action = None
        self.current_log_prob = None
        self.current_value = None
        self.current_ah = None
        self.current_ch = None
        self.episode_reward = 0.0

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
        except:
            return
        if msg.get('type') == 'step':
            self.handle_step(msg, addr)
        elif msg.get('type') == 'episode_end':
            self.handle_episode_end(msg, addr)

    def handle_step(self, msg, addr) -> None:
        inputs = msg.get('inputs', [0.0]*INPUT_DIM)
        reward = msg.get('reward', 0.0)
        self.episode_reward += reward
        
        if self.current_state is not None:
            self.agent.store_transition(
                state=self.current_state,
                action=self.current_action,
                reward=reward,
                done=False,
                log_prob=self.current_log_prob,
                value=self.current_value,
                ah=self.current_ah,
                ch=self.current_ch
            )

        # Obtener acción para el estado actual
        action, log_p, val, ah, ch = self.agent.get_action(inputs)

        # Actualizar estado actual
        self.current_state = inputs
        self.current_action = action
        self.current_log_prob = log_p
        self.current_value = val
        self.current_ah = ah
        self.current_ch = ch

        response = {
            'move_x': action['move_x'],
            'move_y': action['move_y'],
            'shot_angle': action['shot_angle'],
            'action': action['action'],
            'move_dir': [action['move_x'], action['move_y']]
        }
        self.sock.sendto(json.dumps(response).encode('utf-8'), addr)

    def handle_episode_end(self, msg, addr) -> None:
        final_reward = msg.get('reward', 0.0)
        timed_out = msg.get('timed_out', False)
        self.episode_reward += final_reward
        
        last_state_for_bootstrap = self.current_state

        
        if self.current_state is not None:
            self.agent.store_transition(
                state=self.current_state,
                action=self.current_action,
                reward=final_reward,
                done=True,
                log_prob=self.current_log_prob,
                value=self.current_value,
                ah=self.current_ah,
                ch=self.current_ch
            )
            self.current_state = None
            self.current_action = None
            self.current_ah = None
            self.current_ch = None
            self.current_log_prob = None
            self.current_value = None
        
        self.agent.end_episode(self.episode_reward, last_state_for_bootstrap, timed_out)
        self.episode_reward = 0.0
        ack = {'type': 'episode_ready'}
        self.sock.sendto(json.dumps(ack).encode('utf-8'), addr)

    def stop(self) -> None:
        self.running = False
        self.sock.close()

# ------------------------------
# Punto de entrada
# ------------------------------
if __name__ == "__main__":
    config = {
        'input_dim': INPUT_DIM,
        'discrete_actions': DISCRETE_ACTIONS,
        'hidden_size': HIDDEN_SIZE,
        'lr': LR,
        'gamma': GAMMA,
        'gae_lambda': GAE_LAMBDA,
        'clip_eps': CLIP_EPS,
        'entropy_coef': ENTROPY_COEF,
        'value_coef': VALUE_COEF,
        'max_grad_norm': MAX_GRAD_NORM,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'buffer_size': BUFFER_SIZE,
        'model_save_path': MODEL_SAVE_PATH,
        'model_load_path': MODEL_LOAD_PATH
    }
    agent = PPOAgent(config)
    server = GodotRLServer(agent, HOST, PORT)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nApagando servidor...")
        server.stop()
        agent.save_model(config['model_save_path'])
        print("Servidor detenido.")