import numpy as np
import json
import threading
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import time

# ---------- Hiperparámetros ----------
INPUTS = 40
HIDDEN = 256
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
        self.shared = nn.Sequential(
            nn.Linear(INPUTS, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(HIDDEN, 3)      # salidas continuas (tanh internamente)
        self.actor_logstd = nn.Parameter(torch.zeros(3))  # log std desacoplado
        self.actor_shoot = nn.Linear(HIDDEN, 1)     # logit para disparar (Bernoulli)
        self.critic = nn.Linear(HIDDEN, 1)

        # Inicialización
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_shoot.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x) -> any:
        # x: tensor (batch, INPUTS)
        features = self.shared(x)
        # Acciones continuas
        mean = torch.tanh(self.actor_mean(features))  # rango [-1, 1]
        log_std = self.actor_logstd.clamp(-5, 2)
        std = log_std.exp()
        # Acción discreta
        shoot_logit = self.actor_shoot(features).squeeze(-1)
        # Valor
        value = self.critic(features).squeeze(-1)
        return mean, std, shoot_logit, value

    def get_action(self, x, deterministic=False) -> any:
        mean, std, shoot_logit, value = self.forward(x)
        if deterministic:
            move = mean
            shoot = (shoot_logit > 0).float()
        else:
            # Exploración: muestrear de la propia distribución N(mean, std)
            noise = torch.randn_like(mean) * std
            move = (mean + noise).clamp(-1, 1)
            prob = torch.sigmoid(shoot_logit)
            shoot = torch.bernoulli(prob).float()
        # Calcular log_prob con la misma std (consistente)
        log_prob_cont = -0.5 * (((move - mean) / (std + 1e-8)) ** 2 + 2 * std.log() + np.log(2 * np.pi)).sum(dim=-1)
        p = torch.sigmoid(shoot_logit)
        log_prob_shoot = shoot * p.log() + (1 - shoot) * (1 - p).log()
        log_prob = log_prob_cont + log_prob_shoot
        return move, shoot, log_prob, value

    def evaluate(self, x, actions) -> any:
        """Calcula log_prob y valor para acciones dadas (necesario para PPO update)"""
        mean, std, shoot_logit, value = self.forward(x)
        move_act = actions[:, :3]
        shoot_act = actions[:, 3].long()
        # Log prob continua
        log_prob_cont = -0.5 * (((move_act - mean) / std) ** 2 + 2 * std.log() + np.log(2 * np.pi)).sum(dim=-1)
        # Log prob discreta
        p = torch.sigmoid(shoot_logit)
        log_prob_shoot = shoot_act * p.log() + (1 - shoot_act) * (1 - p).log()
        log_prob = log_prob_cont + log_prob_shoot
        # Entropía (para regularización)
        dim = 3
        entropy_cont = 0.5 * (dim * (1 + np.log(2 * np.pi)) + (2 * std.log()).sum(dim=-1))
        entropy_discrete = -(p * p.log() + (1 - p) * (1 - p).log())
        entropy = (entropy_cont + entropy_discrete).mean()
        return log_prob, value, entropy

# ---------- Cliente servidor UDP ----------
class PPOServer:
    def __init__(self, host="127.0.0.1", port=9999) -> None:
        self.best_model_path = "./assets/train_data/best_boss_brain.json"
        self.train_data_path = "./assets/train_data/train_data.json"
        self.training_log_path = "./assets/train_data/training_log.json"
        self.load_training_state()
        self.best_avg_reward = -1e9  # se cargará del archivo
        self.net = PPOActorCritic().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.host = host
        self.port = port
        self.episode_buffer = []   # guarda (state, action, old_log_prob, reward, done, value)
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
        print(f"[server] listening on {host}:{port}")

    def process_step(self, msg) -> dict:
        x = torch.tensor(msg["inputs"], dtype=torch.float32, device=device).unsqueeze(0)
        reward = msg.get("reward", 0.0)

        with torch.no_grad():
            move, shoot, log_prob, value = self.net.get_action(x)
            move = move.squeeze(0).cpu().numpy()
            shoot = shoot.item()
            log_prob = log_prob.item()
            value = value.item()

        # Guardar transición para GAE (si existe el estado anterior)
        if self.last_state is not None:
            self.episode_buffer.append({
                "state": self.last_state,
                "action": self.last_action,
                "old_log_prob": self.last_log_prob,
                "reward": reward,
                "done": False,
                "value": self.last_value
            })

        self.last_state = x
        self.last_action = torch.tensor([move[0], move[1], move[2], shoot], dtype=torch.float32, device=device).unsqueeze(0)
        self.last_log_prob = log_prob
        self.last_value = value

        return {
            "move_dir": [float(move[0]), float(move[1])],
            "shot_angle": float(move[2]),
            "action": int(shoot)
        }

    def process_episode_end(self, msg) -> None:
        final_reward = msg.get("reward", 0.0)
        episode_num = msg.get("episode", 0)
        total_reward = msg.get("total_reward", 0.0)

        # Agregar la última transición con reward terminal y done=True
        if self.last_state is not None:
            self.episode_buffer.append({
                "state": self.last_state,
                "action": self.last_action,
                "old_log_prob": self.last_log_prob,
                "reward": final_reward,
                "done": True,
                "value": self.last_value
            })

        loss = 0.0
        advantages = None
        returns = None
        steps = len(self.episode_buffer)

        if steps > 1:
            advantages, returns, loss = self._update_policy()
        else:
            advantages, returns, loss = None, None, 0.0
        
        self.save_model()
        self.check_and_save_best_model(total_reward, episode_num)

        # Guardar estadísticas detalladas
        if advantages is not None:
            self._log_training_stats(episode_num, total_reward, final_reward, len(self.episode_buffer), advantages, returns, loss)
        

        # Limpiar buffer y resetear estados
        self.episode_buffer.clear()
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None

        print(f"[server] episode {episode_num} | total_reward={total_reward:.2f} | final={final_reward:.2f} | steps={len(self.episode_buffer)}")

    def _update_policy(self) -> any:
        # Preparar tensores desde el buffer
        states = torch.cat([t["state"] for t in self.episode_buffer], dim=0)
        actions = torch.cat([t["action"] for t in self.episode_buffer], dim=0)
        old_log_probs = torch.tensor([t["old_log_prob"] for t in self.episode_buffer], device=device)
        rewards = torch.tensor([t["reward"] for t in self.episode_buffer], dtype=torch.float32, device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        dones = torch.tensor([t["done"] for t in self.episode_buffer], dtype=torch.float32, device=device)
        values = torch.tensor([t["value"] for t in self.episode_buffer], dtype=torch.float32, device=device)

        # Calcular GAE y retornos
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if dones[t] else (values[t+1] if t+1 < len(values) else 0.0)
            delta = rewards[t] + GAMMA * next_val - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        # Normalizar advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Múltiples épocas (usamos el episodio completo como un batch)
        for _ in range(EPOCHS):
            # Evaluar con política actual
            log_probs, new_values, entropy = self.net.evaluate(states, actions)
            new_values = new_values.squeeze(-1)

            # Ratio de importancia
            ratio = (log_probs - old_log_probs).exp()
            # Pérdida actor con clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            # Pérdida critic (MSE)
            critic_loss = nn.MSELoss()(new_values, returns)
            # Pérdida total
            loss = actor_loss + 0.5 * critic_loss - ENTROPY_B * entropy

            self.optimizer.zero_grad() 
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
        return advantages, returns, loss.item()

    def load_training_state(self) -> None:
        """Carga el progreso desde train_data.json (si existe)"""
        if os.path.exists(self.train_data_path):
            with open(self.train_data_path, "r") as f:
                data = json.load(f)
            self.best_avg_reward = data.get("best_avg_reward", -1e9)
            self.last_episode = data.get("episode", 0)
            print(f"[server] loaded training state: episode {self.last_episode}, best_avg_reward={self.best_avg_reward:.2f}")
        else:
            self.best_avg_reward = -1e9
            self.last_episode = 0
            print("[server] no existing training data, starting fresh")
    
    def check_and_save_best_model(self, total_reward, episode) -> None:
        """Guarda el modelo si la recompensa total supera la mejor registrada"""
        if total_reward > self.best_avg_reward:
            self.best_avg_reward = total_reward
            self.save_model(self.best_model_path)
            print(f"[server] 🏆 new best model saved (reward={total_reward:.2f}) at episode {episode}")
    
    def _log_training_stats(self, episode, total_reward, final_reward, steps, advantages, returns, loss) -> None:
        """Registra estadísticas detalladas del entrenamiento"""
        log_entry = {
            "episode": episode,
            "total_reward": total_reward,
            "final_reward": final_reward,
            "steps": steps,
            "mean_advantage": float(advantages.detach().cpu().numpy().mean()) if advantages is not None and advantages.numel() > 0 else 0.0,
            "mean_return": float(returns.detach().cpu().numpy().mean()) if returns is not None and returns.numel() > 0 else 0.0,
            "loss": float(loss),
            "timestamp": time.time()
        }
        # Cargar log existente
        if os.path.exists(self.training_log_path):
            with open(self.training_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        else:
            log = []
        log.append(log_entry)
        with open(self.training_log_path, "w") as f:
            json.dump(log, f, indent=2)
    
    def save_model(self, path="./assets/train_data/boss_brain.json") -> None:
        # Convertir a dict de numpy para guardar en JSON (compatible con el juego)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = self.net.state_dict()
        np_state = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
        with open(path, "w") as f:
            json.dump(np_state, f)

    def load_model(self, path="./assets/train_data/boss_brain.json") -> None:
        if os.path.exists(path):
            with open(path, "r") as f:
                np_state = json.load(f)
            state_dict = {k: torch.tensor(v, device=device) for k, v in np_state.items()}
            self.net.load_state_dict(state_dict)
            print(f"[server] loaded model from {path}")
        else:
            print("[server] starting with fresh model")

    def run(self) -> None:
        if run_best_model:
            self.load_training_state()
        else:
            self.load_model()
        while True:
            data, addr = self.sock.recvfrom(65535)
            msg = json.loads(data.decode())
            if msg["type"] == "step":
                resp = self.process_step(msg)
                self.sock.sendto(json.dumps(resp).encode(), addr)
            elif msg["type"] == "episode_end":
                self.process_episode_end(msg)

if __name__ == "__main__":
    server = PPOServer()
    server.run()