import copy
import queue

import numpy as np
import json
import threading
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

# ---------- Hiperparámetros ----------
INPUTS = 40
HIDDEN = 256
CHUNK_LEN = 64         # largo de cada sub-secuencia para mini-batches
ACTOR_OUT = 4          # 3 continuos (move_x, move_y, angle) + 1 logit para disparar
GAMMA = 0.99
LAMBDA = 0.95
LR = 3e-4
CLIP_EPS = 0.2
ENTROPY_B = 0.01
EPOCHS = 4
BATCH_SIZE = 64        # no se usa realmente, procesamos el episodio completo
MAX_GRAD_NORM = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_best_model = False

# ---------- Definición de la red (PPO) ----------
class PPOActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # --- Trunk del Actor ---
        self.actor_trunk = nn.Sequential(
            nn.Linear(INPUTS, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.Tanh()
        )
        self.actor_lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)

        # Cabezas del actor
        self.actor_mean    = nn.Linear(HIDDEN, 3)   # move_x, move_y, shot_angle
        self.actor_logstd  = nn.Parameter(torch.full((3,), -0.5))  # std inicial más conservadora
        self.actor_shoot   = nn.Linear(HIDDEN, 1)   # logit disparo (Bernoulli)

        # --- Trunk del Crítico ---
        self.critic_trunk = nn.Sequential(
            nn.Linear(INPUTS, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),   # capa extra — el crítico necesita más capacidad
            nn.LayerNorm(HIDDEN),
            nn.Tanh()
        )
        self.critic_lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
        self.critic_head  = nn.Linear(HIDDEN, 1)

        # Inicialización ortogonal
        self._init_weights()

    def _init_weights(self) -> None:
        for module in list(self.actor_trunk) + list(self.critic_trunk):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # LSTMs: inicialización ortogonal en las matrices de pesos
        for lstm in [self.actor_lstm, self.critic_lstm]:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
                    # Bias de forget gate a 1 — ayuda a retener memoria al inicio
                    n = param.size(0)
                    param.data[n//4 : n//2].fill_(1.0)

        # Cabezas del actor: gain pequeño para acciones iniciales casi uniformes
        nn.init.orthogonal_(self.actor_mean.weight,  gain=0.01)
        nn.init.constant_(self.actor_mean.bias,      0.0)
        nn.init.orthogonal_(self.actor_shoot.weight, gain=0.01)
        nn.init.constant_(self.actor_shoot.bias,     0.0)

        # Crítico: gain 1.0
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias,     0.0)

    def make_hidden(self, batch_size: int = 1):
        """Crea hidden states en cero para actor y crítico."""
        zeros = lambda: (
            torch.zeros(1, batch_size, HIDDEN, device=device),
            torch.zeros(1, batch_size, HIDDEN, device=device)
        )
        return zeros(), zeros()   # (actor_hidden, critic_hidden)
    
    def forward_actor(self, x, hidden) -> any:
        """
        x:      (batch, seq_len, INPUTS)  o  (batch, INPUTS) → se unsqueeze internamente
        hidden: (h, c) del LSTM actor
        Devuelve mean, std, shoot_logit, nuevo hidden
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (batch, 1, INPUTS)
        features = self.actor_trunk(x)   # (batch, seq, HIDDEN)
        lstm_out, new_hidden = self.actor_lstm(features, hidden)
        out = lstm_out[:, -1, :]         # último step de la secuencia

        mean        = torch.tanh(self.actor_mean(out))
        log_std     = self.actor_logstd.clamp(-5, 2)
        std         = log_std.exp()
        shoot_logit = self.actor_shoot(out).squeeze(-1)
        return mean, std, shoot_logit, new_hidden
    
    def forward_critic(self, x, hidden) -> any:
        """
        x:      (batch, seq_len, INPUTS)  o  (batch, INPUTS)
        hidden: (h, c) del LSTM crítico
        Devuelve value, nuevo hidden
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.critic_trunk(x)
        lstm_out, new_hidden = self.critic_lstm(features, hidden)
        out   = lstm_out[:, -1, :]
        value = self.critic_head(out).squeeze(-1)
        return value, new_hidden

    def get_action(self, x, actor_hidden, critic_hidden, deterministic=False) -> any:
        """
        Usado en inferencia (process_step).
        x: tensor (1, INPUTS) — un solo step
        """
        mean, std, shoot_logit, new_actor_h = self.forward_actor(x, actor_hidden)

        if deterministic:
            move  = mean
            shoot = (shoot_logit > 0).float()
        else:
            noise = torch.randn_like(mean) * std
            move  = (mean + noise).clamp(-1, 1)
            prob  = torch.sigmoid(shoot_logit)
            shoot = torch.bernoulli(prob)

        # Log prob de la acción tomada
        log_prob_cont  = -0.5 * (
            ((move - mean) / (std + 1e-8)) ** 2
            + 2 * std.log()
            + np.log(2 * np.pi)
        ).sum(dim=-1)
        p = torch.sigmoid(shoot_logit)
        log_prob_shoot = shoot * p.log() + (1 - shoot) * (1 - p).log()
        log_prob = log_prob_cont + log_prob_shoot

        value, new_critic_h = self.forward_critic(x, critic_hidden)

        return move, shoot, log_prob, value, new_actor_h, new_critic_h

    def evaluate(self, states, actions, actor_hiddens_0, critic_hiddens_0) -> any:
        """
        Usado en el update PPO. Recalcula log_probs y values propagando el LSTM
        sobre toda la secuencia del chunk.

        states:          (batch, seq_len, INPUTS)
        actions:         (batch, seq_len, 4)
        actor_hiddens_0: (h0, c0) del inicio del chunk — shape (1, batch, HIDDEN)
        critic_hiddens_0: ídem para crítico
        """
        # --- Actor ---
        features_a = self.actor_trunk(states)                          # (B, T, HIDDEN)
        lstm_out_a, _ = self.actor_lstm(features_a, actor_hiddens_0)   # (B, T, HIDDEN)

        mean        = torch.tanh(self.actor_mean(lstm_out_a))          # (B, T, 3)
        log_std     = self.actor_logstd.clamp(-5, 2)
        std         = log_std.exp()
        shoot_logit = self.actor_shoot(lstm_out_a).squeeze(-1)         # (B, T)

        move_act  = actions[:, :, :3]                                  # (B, T, 3)
        shoot_act = actions[:, :, 3].long()                            # (B, T)

        log_prob_cont = -0.5 * (
            ((move_act - mean) / (std + 1e-8)) ** 2
            + 2 * std.log()
            + np.log(2 * np.pi)
        ).sum(dim=-1)                                                   # (B, T)

        p = torch.sigmoid(shoot_logit)
        log_prob_shoot = shoot_act * p.log() + (1 - shoot_act) * (1 - p).log()
        log_probs = log_prob_cont + log_prob_shoot                      # (B, T)

        # Entropía
        dim = 3
        entropy_cont    = 0.5 * (dim * (1 + np.log(2 * np.pi)) + (2 * std.log()).sum(dim=-1))
        entropy_disc    = -(p * p.log() + (1 - p) * (1 - p).log())
        entropy         = (entropy_cont + entropy_disc).mean()

        # --- Crítico ---
        features_c = self.critic_trunk(states)
        lstm_out_c, _ = self.critic_lstm(features_c, critic_hiddens_0)
        values = self.critic_head(lstm_out_c).squeeze(-1)               # (B, T)

        return log_probs, values, entropy

# ---------- Cliente servidor UDP ----------
class PPOServer:
    def __init__(self, host="127.0.0.1", port=9999) -> None:
        self.best_model_path  = "./assets/train_data/best_boss_brain.json"
        self.train_data_path  = "./assets/train_data/train_data.json"
        self.training_log_path = "./assets/train_data/training_log.json"
        self.load_training_state()
        self.best_avg_reward = -1e9

        self.net       = PPOActorCritic().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))

        # Buffer del episodio — cada entrada es un dict con todo lo necesario
        self.episode_buffer = []

        # Hidden states actuales (se propagan step a step)
        self.actor_hidden  = None
        self.critic_hidden = None

        # Último step guardado (para poder asociar reward al step anterior)
        self.last_state        = None
        self.last_action       = None
        self.last_log_prob     = None
        self.last_value        = None
        self.last_actor_hidden = None   # hidden al INICIO del último step
        self.last_critic_hidden = None

        self.notify_queue = queue.Queue()
        self.file_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.sock_lock = threading.Lock()

        print(f"[server] Escuchando.. {host}:{port}")
        self._reset_hidden()

    # ----------------------------------------
    # --- resetea los hidden states a cero ---
    # ----------------------------------------
    def _reset_hidden(self) -> None:
        self.actor_hidden, self.critic_hidden = self.net.make_hidden(batch_size=1)

    # --------------------------------------
    # --- Inferencia (un step del juego) ---
    # --------------------------------------
    def process_step(self, msg) -> dict:
        x      = torch.tensor(msg["inputs"], dtype=torch.float32, device=device).unsqueeze(0)
        reward = msg.get("reward", 0.0)

        # Guardamos el hidden ANTES de pasarlo por la red — es el contexto
        # que tenía la red cuando tomó la acción del step anterior
        prev_actor_h  = (self.actor_hidden[0].detach().clone(),
                         self.actor_hidden[1].detach().clone())
        prev_critic_h = (self.critic_hidden[0].detach().clone(),
                         self.critic_hidden[1].detach().clone())

        with self.model_lock:
            with torch.no_grad():
                move, shoot, log_prob, value, new_actor_h, new_critic_h = \
                    self.net.get_action(x, self.actor_hidden, self.critic_hidden)

            move      = move.squeeze(0).cpu().numpy()
            shoot     = shoot.item()
            log_prob  = log_prob.item()
            value     = value.item()

        # Actualizamos los hidden states para el próximo step
        self.actor_hidden  = (new_actor_h[0].detach(), new_actor_h[1].detach())
        self.critic_hidden = (new_critic_h[0].detach(), new_critic_h[1].detach())

        # Guardamos la transición del step ANTERIOR (ahora ya tenemos su reward)
        if self.last_state is not None:
            self.episode_buffer.append({
                "state":         self.last_state,               # (1, INPUTS)
                "action":        self.last_action,              # (1, 4)
                "old_log_prob":  self.last_log_prob,            # float
                "reward":        reward,                        # float
                "done":          False,
                "value":         self.last_value,               # float
                # Hiddens al inicio del step — necesarios para re-propagar en update
                "actor_h0":      self.last_actor_hidden[0],    # (1, 1, HIDDEN)
                "actor_c0":      self.last_actor_hidden[1],
                "critic_h0":     self.last_critic_hidden[0],
                "critic_c0":     self.last_critic_hidden[1],
            })

        # Guardamos el estado actual como "último" para el próximo step
        action_tensor = torch.tensor(
            [move[0], move[1], move[2], shoot],
            dtype=torch.float32, device=device
        ).unsqueeze(0)

        self.last_state         = x
        self.last_action        = action_tensor
        self.last_log_prob      = log_prob
        self.last_value         = value
        self.last_actor_hidden  = prev_actor_h
        self.last_critic_hidden = prev_critic_h

        return {
            "move_dir":   [float(move[0]), float(move[1])],
            "shot_angle": float(move[2]),
            "action":     int(shoot)
        }

    def process_episode_end(self, msg, client_addr) -> None:
        final_reward = msg.get("reward", 0.0)
        episode_num  = msg.get("episode", 0)
        total_reward = msg.get("total_reward", 0.0)

        # Guardamos la última transición con done=True y reward terminal
        if self.last_state is not None:
            self.episode_buffer.append({
                "state":         self.last_state,
                "action":        self.last_action,
                "old_log_prob":  self.last_log_prob,
                "reward":        final_reward,
                "done":          True,
                "value":         self.last_value,
                "actor_h0":      self.last_actor_hidden[0],
                "actor_c0":      self.last_actor_hidden[1],
                "critic_h0":     self.last_critic_hidden[0],
                "critic_c0":     self.last_critic_hidden[1],
            })

        # Reset completo para el próximo episodio
        buffer_copy = copy.deepcopy(self.episode_buffer)
        self.episode_buffer.clear()
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
        self.last_actor_hidden = None
        self.last_critic_hidden = None
        self._reset_hidden()
        
        # Si hay suficientes pasos, entrenar en hilo
        if len(buffer_copy) > CHUNK_LEN:
            threading.Thread(
                target=self._training_worker,
                args=(buffer_copy, episode_num, total_reward, final_reward, client_addr),
                daemon=True
            ).start()
        else:
            # Sin entrenamiento, responder inmediatamente
            self.sock.sendto(json.dumps({"type": "episode_ready"}).encode(), client_addr)
            print(f"[server] ep {episode_num} (sin entrenamiento) | reward={total_reward:.2f} | steps={len(buffer_copy)}")

    def _update_policy_from_buffer(self, buffer) -> tuple:
        T = len(buffer)

        # Construir tensores
        states = torch.cat([t["state"] for t in buffer], dim=0)
        actions = torch.cat([t["action"] for t in buffer], dim=0)
        old_lp = torch.tensor([t["old_log_prob"] for t in buffer], device=device)
        rewards_raw = torch.tensor([t["reward"] for t in buffer], dtype=torch.float32, device=device)
        dones = torch.tensor([t["done"] for t in buffer], dtype=torch.float32, device=device)
        values_old = torch.tensor([t["value"] for t in buffer], dtype=torch.float32, device=device)

        # Normalizar rewards
        if rewards_raw.std() < 1e-8:
            rewards = rewards_raw - rewards_raw.mean()
        else:
            rewards = (rewards_raw - rewards_raw.mean()) / (rewards_raw.std() + 1e-8)

        # GAE
        advantages = torch.zeros(T, device=device)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = 0.0 if dones[t] else (values_old[t + 1].item() if t + 1 < T else 0.0)
            delta = rewards[t] + GAMMA * next_val - values_old[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values_old
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Dividir en chunks
        n_chunks = max(1, T // CHUNK_LEN)
        chunk_len = CHUNK_LEN if T >= CHUNK_LEN else T
        T_used = n_chunks * chunk_len
        chunk_starts = list(range(0, T_used, chunk_len))

        last_loss = 0.0
        for _ in range(EPOCHS):
            perm = torch.randperm(n_chunks).tolist()
            for chunk_idx in perm:
                start = chunk_starts[chunk_idx]
                end = start + chunk_len

                s_chunk = states[start:end].unsqueeze(0)
                a_chunk = actions[start:end].unsqueeze(0)
                lp_chunk = old_lp[start:end]
                adv_chunk = advantages[start:end]
                ret_chunk = returns[start:end]

                a_h0 = (buffer[start]["actor_h0"].detach(), buffer[start]["actor_c0"].detach())
                c_h0 = (buffer[start]["critic_h0"].detach(), buffer[start]["critic_c0"].detach())

                with self.model_lock:
                    log_probs_seq, values_seq, entropy = self.net.evaluate(s_chunk, a_chunk, a_h0, c_h0)

                    log_probs_seq = log_probs_seq.squeeze(0)
                    values_seq = values_seq.squeeze(0)

                    ratio = (log_probs_seq - lp_chunk).exp()
                    surr1 = ratio * adv_chunk
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_chunk
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.functional.mse_loss(values_seq, ret_chunk)
                    loss = actor_loss + 0.5 * critic_loss - ENTROPY_B * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
                    self.optimizer.step()
                    last_loss = loss.item()

        return advantages, returns, last_loss
    
    def _training_worker(self, buffer, episode_num, total_reward, final_reward, client_addr) -> None:
        try:
            # Actualizar la política usando el buffer copiado
            advantages, returns, loss = self._update_policy_from_buffer(buffer)

            # Guardar modelo (con lock)
            with self.model_lock:
                self.save_model()
                self.check_and_save_best_model(total_reward, episode_num)

            # Registrar estadísticas
            if advantages is not None:
                self._log_training_stats(episode_num, total_reward, final_reward,
                                         len(buffer), advantages, returns, loss)

            print(f"[server] ep {episode_num} (entrenado) | reward={total_reward:.2f} | final={final_reward:.2f} | steps={len(buffer)} | loss={loss:.4f}")
        except Exception as e:
            print(f"[server] Error en entrenamiento del episodio {episode_num}: {e}")
        finally:
            # Notificar al cliente que puede continuar
            self.notify_queue.put(client_addr)

    # -------------------------------------------
    # --- Actualizacion y Control de archivos ---
    # -------------------------------------------
    def load_training_state(self) -> None:
        if os.path.exists(self.train_data_path):
            with open(self.train_data_path, "r") as f:
                data = json.load(f)
            self.best_avg_reward = data.get("best_avg_reward", -1e9)
            self.last_episode    = data.get("episode", 0)
            print(f"[server] training state cargado: ep {self.last_episode}, best={self.best_avg_reward:.2f}")
        else:
            self.best_avg_reward = -1e9
            self.last_episode    = 0
            print("[server] sin datos previos, arrancando desde cero")

    def check_and_save_best_model(self, total_reward, episode) -> None:
        if total_reward > self.best_avg_reward:
            self.best_avg_reward = total_reward
            self.save_model(self.best_model_path)
            print(f"[server] nuevo mejor modelo (reward={total_reward:.2f}) ep {episode}")

    def _log_training_stats(self, episode, total_reward, final_reward, steps, advantages, returns, loss) -> None:
        entry = {
            "episode":        episode,
            "total_reward":   total_reward,
            "final_reward":   final_reward,
            "steps":          steps,
            "mean_advantage": float(advantages.detach().cpu().mean()) if advantages is not None else 0.0,
            "mean_return":    float(returns.detach().cpu().mean())    if returns    is not None else 0.0,
            "loss":           float(loss),
            "timestamp":      time.time()
        }
        with self.file_lock:
            with open(self.training_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def save_model(self, path="./assets/train_data/boss_brain.json") -> None:
        with self.file_lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np_state = {k: v.cpu().numpy().tolist() for k, v in self.net.state_dict().items()}
            with open(path, "w") as f:
                json.dump(np_state, f)

    def load_model(self, path="./assets/train_data/boss_brain.json") -> None:
        if os.path.exists(path):
            with open(path, "r") as f:
                np_state = json.load(f)
            state_dict = {k: torch.tensor(v, device=device) for k, v in np_state.items()}
            self.net.load_state_dict(state_dict)
            print(f"[server] modelo cargado desde {path}")
        else:
            print("[server] sin modelo previo, pesos aleatorios")

    def run(self) -> None:
        self.load_model()
        while True:
            # Notificaciones pendientes de workers
            try:
                addr = self.notify_queue.get_nowait()
                self.sock.sendto(json.dumps({"type": "episode_ready"}).encode(), addr)
            except queue.Empty:
                pass

            try:
                data, addr = self.sock.recvfrom(65535)
            except (socket.timeout, BlockingIOError):
                continue

            msg = json.loads(data.decode())
            if msg["type"] == "step":
                resp = self.process_step(msg)
                self.sock.sendto(json.dumps(resp).encode(), addr)
            elif msg["type"] == "episode_end":
                self.process_episode_end(msg, addr)

if __name__ == "__main__":
    server = PPOServer()
    server.run()