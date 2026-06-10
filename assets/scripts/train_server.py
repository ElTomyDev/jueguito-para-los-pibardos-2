import numpy as np
import json
from collections import deque
import threading
import queue as queue_module
import socket

# Hiperparámetros 
INPUTS      = 40
HIDDEN      = 256
ACTOR_OUT   = 4
GAMMA       = 0.99
LAMBDA      = 0.95      # GAE lambda
LR          = 0.0003
CLIP_EPS    = 0.2       # clip de PPO
ENTROPY_B   = 0.01
MAX_GRAD    = 0.5
EPOCHS      = 4         # épocas por batch en PPO
BATCH_SIZE  = 256
MINI_BATCH  = 64

def sigmoid(x) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def clip_grad(g) -> any:
    return np.clip(g, -MAX_GRAD, MAX_GRAD)

class LSTMCell:
    """LSTM minimal implementado en numpy"""
    def __init__(self, input_size, hidden_size):
        s = np.sqrt(2.0 / (input_size + hidden_size))
        # Pesos para los 4 gates: input, forget, cell, output
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * s
        self.bf = np.ones(hidden_size)   # forget bias en 1 para estabilidad inicial

        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * s
        self.bi = np.zeros(hidden_size)

        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * s
        self.bc = np.zeros(hidden_size)

        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * s
        self.bo = np.zeros(hidden_size)

        self.hidden_size = hidden_size

    def forward(self, x, h_prev, c_prev):
        """
        x: input vector (input_size,)
        h_prev, c_prev: estados anteriores (hidden_size,)
        Devuelve h_new, c_new y cache para backprop
        """
        xh = np.concatenate([x, h_prev])

        f = sigmoid(self.Wf @ xh + self.bf)        # forget gate
        i = sigmoid(self.Wi @ xh + self.bi)        # input gate
        c_tilde = np.tanh(self.Wc @ xh + self.bc) # candidate
        o = sigmoid(self.Wo @ xh + self.bo)        # output gate

        c_new = f * c_prev + i * c_tilde
        h_new = o * np.tanh(c_new)

        cache = {
            "x": x, "h_prev": h_prev, "c_prev": c_prev,
            "xh": xh, "f": f, "i": i,
            "c_tilde": c_tilde, "o": o,
            "c_new": c_new, "h_new": h_new
        }
        return h_new, c_new, cache

    def zero_state(self):
        return np.zeros(self.hidden_size), np.zeros(self.hidden_size)

    def backward(self, dh, dc, cache):
        xh      = cache["xh"]

        f       = cache["f"]
        i       = cache["i"]
        o       = cache["o"]

        c_prev  = cache["c_prev"]
        c_tilde = cache["c_tilde"]
        c_new   = cache["c_new"]

        tanh_c = np.tanh(c_new)

        do = dh * tanh_c
        dc = dc + dh * o * (1.0 - tanh_c * tanh_c)

        df = dc * c_prev
        di = dc * c_tilde
        dg = dc * i

        dc_prev = dc * f

        df_raw = df * f * (1.0 - f)
        di_raw = di * i * (1.0 - i)
        do_raw = do * o * (1.0 - o)
        dg_raw = dg * (1.0 - c_tilde * c_tilde)

        gWf = np.outer(df_raw, xh)
        gWi = np.outer(di_raw, xh)
        gWo = np.outer(do_raw, xh)
        gWc = np.outer(dg_raw, xh)

        gbf = df_raw
        gbi = di_raw
        gbo = do_raw
        gbc = dg_raw

        dxh = (
            self.Wf.T @ df_raw +
            self.Wi.T @ di_raw +
            self.Wo.T @ do_raw +
            self.Wc.T @ dg_raw
        )

        return {
            "dxh": dxh,
            "dc_prev": dc_prev,

            "Wf": gWf,
            "Wi": gWi,
            "Wo": gWo,
            "Wc": gWc,

            "bf": gbf,
            "bi": gbi,
            "bo": gbo,
            "bc": gbc
        }

class PPOActorCritic:

    def __init__(self):
        s1 = np.sqrt(2.0 / INPUTS)
        s2 = np.sqrt(2.0 / HIDDEN)

        # Capa de entrada
        self.W_in = np.random.randn(HIDDEN, INPUTS) * s1
        self.b_in = np.zeros(HIDDEN)

        # LSTM
        self.lstm = LSTMCell(HIDDEN, HIDDEN)

        # Cabezas
        self.W_actor  = np.random.randn(ACTOR_OUT, HIDDEN) * s2
        self.b_actor  = np.zeros(ACTOR_OUT)
        self.W_critic = np.random.randn(1, HIDDEN) * s2
        self.b_critic = np.zeros(1)

        # Log std para outputs continuos (aprendible)
        self.log_std = np.full(3, -0.5)  # move_x, move_y, shot_angle

    def forward(self, x, h, c):
        z = np.maximum(0.0, self.W_in @ x + self.b_in)  # ReLU
        h_new, c_new, cache = self.lstm.forward(z, h, c)

        actor_raw  = self.W_actor  @ h_new + self.b_actor
        value      = float((self.W_critic @ h_new + self.b_critic)[0])

        std        = np.exp(self.log_std)
        means      = np.tanh(actor_raw[:3])
        shoot_prob = sigmoid(actor_raw[3])

        return {
            "h": h_new, "c": c_new,
            "z": z, "lstm_cache": cache,
            "actor_raw": actor_raw,
            "means": means,
            "std": std,
            "shoot_prob": shoot_prob,
            "value": value,
        }

    def sample_action(self, state, epsilon=0.1):
        """Muestrea acción con exploración gaussiana"""
        noise  = np.random.randn(3) * state["std"] * epsilon
        move_x = float(np.clip(state["means"][0] + noise[0], -1, 1))
        move_y = float(np.clip(state["means"][1] + noise[1], -1, 1))
        angle  = float(np.clip(state["means"][2] + noise[2], -1, 1))
        shoot  = 1 if np.random.rand() < state["shoot_prob"] else 0
        return move_x, move_y, angle, shoot

    def log_prob(self, state, actions):
        """Log probabilidad de las acciones tomadas — necesario para PPO"""
        std = state["std"]
        # Continuas: log prob gaussiana
        lp_cont = -0.5 * np.sum(
            ((np.array(actions[:3]) - state["means"]) / (std + 1e-8)) ** 2
            + 2 * np.log(std + 1e-8)
        )
        # Discreta: Bernoulli
        p = np.clip(state["shoot_prob"], 1e-6, 1 - 1e-6)
        lp_shoot = actions[3] * np.log(p) + (1 - actions[3]) * np.log(1 - p)
        return lp_cont + lp_shoot

    def compute_gae(self, rewards, values, dones):
        """
        Generalized Advantage Estimation — mucho mejor que TD(0) simple
        Reduce varianza manteniendo algo de bias, controlado por LAMBDA
        """
        advantages = np.zeros(len(rewards))
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0.0 if dones[t] else values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + GAMMA * next_val - values[t]
            gae   = delta + GAMMA * LAMBDA * (0.0 if dones[t] else gae)
            advantages[t] = gae
        returns = advantages + np.array(values[:len(rewards)])
        # Normalización
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def ppo_update(self, trajectories):
        """
        trajectories: lista de dicts con state, actions, old_log_prob, advantage, return_
        """
        for _ in range(EPOCHS):
            indices = np.random.permutation(len(trajectories))
            for start in range(0, len(trajectories), MINI_BATCH):
                batch = [trajectories[i] for i in indices[start:start + MINI_BATCH]]
                self._update_minibatch(batch)

    def _update_minibatch(self, batch):
        gW_in = np.zeros_like(self.W_in)
        gb_in = np.zeros_like(self.b_in)
        gWa   = np.zeros_like(self.W_actor)
        gba   = np.zeros_like(self.b_actor)
        gWc   = np.zeros_like(self.W_critic)
        gbc   = np.zeros_like(self.b_critic)
        gWf = np.zeros_like(self.lstm.Wf)
        gWi = np.zeros_like(self.lstm.Wi)
        gWc_lstm = np.zeros_like(self.lstm.Wc)
        gWo = np.zeros_like(self.lstm.Wo)

        gbf = np.zeros_like(self.lstm.bf)
        gbi = np.zeros_like(self.lstm.bi)
        gbc_lstm = np.zeros_like(self.lstm.bc)
        gbo = np.zeros_like(self.lstm.bo)
        g_log_std = np.zeros_like(self.log_std)

        for tr in batch:
            s        = tr["state"]
            adv      = tr["advantage"]
            ret      = tr["return_"]
            old_lp   = tr["old_log_prob"]
            actions  = tr["actions"]

            new_lp   = self.log_prob(s, actions)
            ratio    = np.exp(np.clip(new_lp - old_lp, -10, 10))

            # PPO clip loss
            clipped  = np.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            pg_loss  = -min(ratio * adv, clipped)

            # Critic loss
            v_pred   = s["value"]
            v_loss   = 0.5 * (v_pred - ret) ** 2

            # Entropy bonus
            p        = np.clip(s["shoot_prob"], 1e-6, 1 - 1e-6)
            entropy  = -(p * np.log(p) + (1-p) * np.log(1-p))
            entropy += 0.5 * np.sum(np.log(2 * np.pi * np.e * s["std"] ** 2))

            # Gradiente del actor (continuo)
            h   = s["h"]
            d_raw = np.zeros(ACTOR_OUT)
            std = s["std"]
            for idx in range(3):
                a_exec = actions[idx]
                mu = s["means"][idx]
                std = s["std"][idx]

                g_log_std[idx] += (
                    ratio * adv *
                    (((a_exec - mu) ** 2) / (std**2 + 1e-8) - 1.0)
                )

            # Gradiente del actor (discreto) con PPO clip
            d_raw[3] = -ratio * adv * (actions[3] - p)
            d_raw[3] += ENTROPY_B * np.log(p / (1 - p)) * p * (1 - p)

            gWa += np.outer(d_raw, h)
            gba += d_raw

            # Gradiente del critic
            d_val = v_pred - ret
            gWc  += np.outer(np.array([d_val]), h)
            gbc  += d_val

            dh = self.W_actor.T @ d_raw + self.W_critic.T[:, 0] * d_val
            cache = s["lstm_cache"]
            lstm_grads = self.lstm.backward(
                dh,
                np.zeros(HIDDEN),
                cache
            )

            gWf += lstm_grads["Wf"]
            gWi += lstm_grads["Wi"]
            gWc_lstm += lstm_grads["Wc"]
            gWo += lstm_grads["Wo"]

            gbf += lstm_grads["bf"]
            gbi += lstm_grads["bi"]
            gbc_lstm += lstm_grads["bc"]
            gbo += lstm_grads["bo"]

            dz = lstm_grads["dxh"][:HIDDEN]

            dz *= (s["z"] > 0)

            gW_in += np.outer(dz, cache["x"])
            gb_in += dz
        
        

        n = len(batch)
        self.lstm.Wf -= LR * clip_grad(gWf / n)
        self.lstm.Wi -= LR * clip_grad(gWi / n)
        self.lstm.Wc -= LR * clip_grad(gWc_lstm / n)
        self.lstm.Wo -= LR * clip_grad(gWo / n)
        self.lstm.bf -= LR * clip_grad(gbf / n)
        self.lstm.bi -= LR * clip_grad(gbi / n)
        self.lstm.bc -= LR * clip_grad(gbc_lstm / n)
        self.lstm.bo -= LR * clip_grad(gbo / n)
        self.log_std -= LR * clip_grad(g_log_std / n)
        self.log_std = np.clip(self.log_std, -4.0, 2.0)
        self.W_actor  -= LR * clip_grad(gWa / n)
        self.b_actor  -= LR * clip_grad(gba / n)
        self.W_critic[0] -= LR * clip_grad(gWc[0] / n)
        self.b_critic[0] -= LR * clip_grad(gbc[0] / n)
        self.W_in     -= LR * clip_grad(gW_in / n)
        self.b_in     -= LR * clip_grad(gb_in / n)

    def save(self, path="./assets/train_data/boss_brain.json"):
        data = {
            "W_in": self.W_in.tolist(), "b_in": self.b_in.tolist(),
            "W_actor": self.W_actor.tolist(), "b_actor": self.b_actor.tolist(),
            "W_critic": self.W_critic.tolist(), "b_critic": self.b_critic.tolist(),
            "log_std": self.log_std.tolist(),
            "lstm_Wf": self.lstm.Wf.tolist(), "lstm_bf": self.lstm.bf.tolist(),
            "lstm_Wi": self.lstm.Wi.tolist(), "lstm_bi": self.lstm.bi.tolist(),
            "lstm_Wc": self.lstm.Wc.tolist(), "lstm_bc": self.lstm.bc.tolist(),
            "lstm_Wo": self.lstm.Wo.tolist(), "lstm_bo": self.lstm.bo.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path="./assets/train_data/boss_brain.json"):
        try:
            with open(path, "r") as f:
                d = json.load(f)
            self.W_in     = np.array(d["W_in"])
            self.b_in     = np.array(d["b_in"])
            self.W_actor  = np.array(d["W_actor"])
            self.b_actor  = np.array(d["b_actor"])
            self.W_critic = np.array(d["W_critic"])
            self.b_critic = np.array(d["b_critic"])
            self.log_std  = np.array(d["log_std"])
            self.lstm.Wf  = np.array(d["lstm_Wf"])
            self.lstm.bf  = np.array(d["lstm_bf"])
            self.lstm.Wi  = np.array(d["lstm_Wi"])
            self.lstm.bi  = np.array(d["lstm_bi"])
            self.lstm.Wc  = np.array(d["lstm_Wc"])
            self.lstm.bc  = np.array(d["lstm_bc"])
            self.lstm.Wo  = np.array(d["lstm_Wo"])
            self.lstm.bo  = np.array(d["lstm_bo"])
            print("[server] modelo PPO+LSTM cargado")
        except FileNotFoundError:
            print("[server] modelo nuevo")

# -------------------------------------------------------
# SERVIDOR
# -------------------------------------------------------

nn = PPOActorCritic()
nn.load()

# Hidden state del LSTM — se resetea por episodio
h_state, c_state = nn.lstm.zero_state()

# Buffer de trayectorias del episodio actual (para GAE)
episode_buffer = []

# Thread de entrenamiento separado para no bloquear el socket
train_queue = queue_module.Queue(maxsize=4)

def training_worker() -> None:
    while True:
        trajectories = train_queue.get()
        if trajectories is None:
            break
        nn.ppo_update(trajectories)
        train_queue.task_done()

train_thread = threading.Thread(target=training_worker, daemon=True)
train_thread.start()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 9999))
print("[server] escuchando 127.0.0.1:9999")

last_state = None
last_actions = None

while True:
    data, addr = sock.recvfrom(65535)
    msg = json.loads(data.decode())

    if msg["type"] == "step":
        x       = np.array(msg["inputs"], dtype=np.float64)
        epsilon = msg.get("epsilon", 0.1)
        reward  = msg.get("reward", 0.0)

        state = nn.forward(x, h_state, c_state)
        h_state = state["h"]
        c_state = state["c"]

        move_x, move_y, angle, shoot = nn.sample_action(state, epsilon)

        # Guarda transición con log_prob para PPO
        if last_state is not None and last_actions is not None:
            old_lp = nn.log_prob(last_state, last_actions)
            episode_buffer.append({
                "state":        last_state,
                "actions":      last_actions,
                "old_log_prob": old_lp,
                "reward":       reward,
                "done":         False,
            })

        last_state   = state
        last_actions = [move_x, move_y, angle, shoot]

        resp = {
            "move_dir":   [move_x, move_y],
            "shot_angle": angle,
            "action":     shoot,
        }
        sock.sendto(json.dumps(resp).encode(), addr)

    elif msg["type"] == "episode_end":
        final_reward = msg.get("reward", 0.0)

        # Cierra la trayectoria con la recompensa terminal
        if last_state is not None and last_actions is not None:
            old_lp = nn.log_prob(last_state, last_actions)
            episode_buffer.append({
                "state":        last_state,
                "actions":      last_actions,
                "old_log_prob": old_lp,
                "reward":       final_reward,
                "done":         True,
            })

        # Calcula GAE sobre la trayectoria completa del episodio
        if len(episode_buffer) >= 2:
            rewards = [t["reward"] for t in episode_buffer]
            values  = [t["state"]["value"] for t in episode_buffer]
            dones   = [t["done"] for t in episode_buffer]
            advantages, returns = nn.compute_gae(rewards, values, dones)

            trajectories = []
            for i, tr in enumerate(episode_buffer):
                trajectories.append({
                    "state":        tr["state"],
                    "actions":      tr["actions"],
                    "old_log_prob": tr["old_log_prob"],
                    "advantage":    float(advantages[i]),
                    "return_":      float(returns[i]),
                })

            if not train_queue.full():
                train_queue.put_nowait(trajectories)

        nn.save()

        # Reset para el próximo episodio
        episode_buffer.clear()
        last_state   = None
        last_actions = None
        h_state, c_state = nn.lstm.zero_state()  # CRÍTICO: resetea LSTM

        print(
            f"[server] ep {msg.get('episode','?')} "
            f"| total={msg.get('total_reward', 0):.2f} "
            f"| final={final_reward:.2f} "
            f"| traj={len(trajectories) if 'trajectories' in dir() else 0}"
        )