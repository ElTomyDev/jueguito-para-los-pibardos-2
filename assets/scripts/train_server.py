import socket
import json
import numpy as np
from collections import deque

HOST = "127.0.0.1"
PORT = 9999

INPUTS    = 49
HIDDEN    = 128
# Outputs: move_x, move_y, shot_angle, shoot_logit
ACTOR_OUT = 4

GAMMA        = 0.99
LR_ACTOR     = 0.0003
LR_CRITIC    = 0.001
ENTROPY_BETA = 0.01
MAX_GRAD     = 1.0

# Replay buffer para reducir varianza
BUFFER_SIZE  = 512
BATCH_SIZE   = 64

# Ruido de exploración para outputs continuos
MOVE_NOISE  = 0.15
ANGLE_NOISE = 0.10


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def clip_grad(g):
    return np.clip(g, -MAX_GRAD, MAX_GRAD)


class ActorCritic:

    def __init__(self):
        s1 = np.sqrt(2.0 / INPUTS)
        s2 = np.sqrt(2.0 / HIDDEN)

        self.W1       = np.random.randn(HIDDEN, INPUTS)  * s1
        self.b1       = np.zeros(HIDDEN)
        self.W_actor  = np.random.randn(ACTOR_OUT, HIDDEN) * s2
        self.b_actor  = np.zeros(ACTOR_OUT)
        self.W_critic = np.random.randn(1, HIDDEN) * s2
        self.b_critic = np.zeros(1)

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        h  = np.maximum(0.0, z1)

        actor_raw  = self.W_actor @ h + self.b_actor
        move_x     = float(np.tanh(actor_raw[0]))
        move_y     = float(np.tanh(actor_raw[1]))
        shot_angle = float(np.tanh(actor_raw[2]))
        shoot_prob = float(sigmoid(actor_raw[3]))
        value      = float((self.W_critic @ h + self.b_critic)[0])

        return {
            "x": x, "h": h, "z1": z1,
            "actor_raw": actor_raw,
            "move_x": move_x,
            "move_y": move_y,
            "shot_angle": shot_angle,
            "shoot_prob": shoot_prob,
            "value": value,
        }

    def train_on_batch(self, batch):
        """
        batch: lista de dicts con keys:
            state, next_state, reward, done,
            action_shoot (0/1),
            action_move_x, action_move_y, action_angle (valores reales ejecutados)
        """
        total_loss = 0.0

        # Acumuladores de gradiente
        gW1 = np.zeros_like(self.W1)
        gb1 = np.zeros_like(self.b1)
        gWa = np.zeros_like(self.W_actor)
        gba = np.zeros_like(self.b_actor)
        gWc = np.zeros_like(self.W_critic)
        gbc = np.zeros_like(self.b_critic)

        advantages = []
        for tr in batch:
            nv = 0.0 if tr["done"] else tr["next_state"]["value"]
            td = tr["reward"] + GAMMA * nv
            advantages.append(td - tr["state"]["value"])

        # Normalización del advantage (reduce varianza)
        adv_arr = np.array(advantages)
        adv_mean = adv_arr.mean()
        adv_std  = adv_arr.std() + 1e-8
        adv_norm = (adv_arr - adv_mean) / adv_std

        for i, tr in enumerate(batch):
            s   = tr["state"]
            adv = float(np.clip(adv_norm[i], -10.0, 10.0))
            nv  = 0.0 if tr["done"] else tr["next_state"]["value"]
            td  = tr["reward"] + GAMMA * nv
            h   = s["h"]

            # ---- Critic ----
            critic_err = s["value"] - td
            gWc += critic_err * h
            gbc += critic_err

            # ---- Actor: outputs discretos (shoot) ----
            d_raw = np.zeros(ACTOR_OUT)

            p = np.clip(s["shoot_prob"], 1e-6, 1.0 - 1e-6)
            act_shoot = tr["action_shoot"]
            d_raw[3] = -(act_shoot - p) * adv
            # Entropy bonus
            d_raw[3] += ENTROPY_BETA * np.log(p / (1.0 - p)) * p * (1.0 - p)

            # ---- Actor: outputs continuos (move_x, move_y, shot_angle) ----
            # Política gaussiana implícita: log_prob ∝ -(a - mu)^2
            # Gradiente respecto a mu_raw: (a_tanh - tanh(raw)) * adv
            # donde a_tanh es el valor ejecutado (ya en espacio tanh)
            for idx, key in [(0, "action_move_x"), (1, "action_move_y"), (2, "action_angle")]:
                raw_val  = s["actor_raw"][idx]
                tanh_val = float(np.tanh(raw_val))
                a_exec   = tr[key]
                # Clamp para que el target esté en (-1, 1)
                a_exec   = float(np.clip(a_exec, -0.999, 0.999))
                # Derivada de tanh para backprop
                dtanh    = 1.0 - tanh_val ** 2
                d_raw[idx] = -(a_exec - tanh_val) * adv * dtanh

            # ---- Backprop ----
            gWa += np.outer(d_raw, h)
            gba += d_raw

            dh_actor  = self.W_actor.T @ d_raw
            dh_critic = self.W_critic.T[:, 0] * critic_err
            dh        = dh_actor + dh_critic
            dh[s["z1"] <= 0.0] = 0.0

            gW1 += np.outer(dh, s["x"])
            gb1 += dh

            total_loss += abs(critic_err)

        n = len(batch)
        # Clipping y actualización
        self.W_actor  -= LR_ACTOR  * clip_grad(gWa / n)
        self.b_actor  -= LR_ACTOR  * clip_grad(gba / n)
        self.W_critic[0] -= LR_CRITIC * clip_grad(gWc[0] / n)
        self.b_critic[0] -= LR_CRITIC * clip_grad(gbc[0] / n)
        self.W1       -= LR_ACTOR  * clip_grad(gW1 / n)
        self.b1       -= LR_ACTOR  * clip_grad(gb1 / n)

        return total_loss / n

    def save(self, path="./assets/train_data/boss_brain.json"):
        data = {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W_actor": self.W_actor.tolist(), "b_actor": self.b_actor.tolist(),
            "W_critic": self.W_critic.tolist(), "b_critic": self.b_critic.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path="./assets/train_data/boss_brain.json"):
        try:
            with open(path, "r") as f:
                d = json.load(f)
            self.W1       = np.array(d["W1"])
            self.b1       = np.array(d["b1"])
            self.W_actor  = np.array(d["W_actor"])
            self.b_actor  = np.array(d["b_actor"])
            self.W_critic = np.array(d["W_critic"])
            self.b_critic = np.array(d["b_critic"])
            print("[server] modelo cargado")
        except FileNotFoundError:
            print("[server] modelo nuevo")


# --------------------------------------------------
# SERVIDOR
# --------------------------------------------------

nn         = ActorCritic()
nn.load()
replay_buf = deque(maxlen=BUFFER_SIZE)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
print(f"[server] escuchando {HOST}:{PORT}")

last_state       = None
last_action_data = None   # guarda los valores continuos ejecutados también


while True:
    data, addr = sock.recvfrom(65535)
    msg = json.loads(data.decode())

    if msg["type"] == "step":
        x     = np.array(msg["inputs"], dtype=np.float64)
        state = nn.forward(x)

        epsilon = msg.get("epsilon", 0.1)

        # --- Muestreo de acciones con exploración ---
        p_shoot = state["shoot_prob"]
        action_shoot = 1 if np.random.rand() < p_shoot else 0

        # Outputs continuos: valor de la red + ruido gaussiano escalado por epsilon
        move_x     = float(np.clip(state["move_x"]     + np.random.randn() * MOVE_NOISE  * epsilon, -1, 1))
        move_y     = float(np.clip(state["move_y"]     + np.random.randn() * MOVE_NOISE  * epsilon, -1, 1))
        shot_angle = float(np.clip(state["shot_angle"] + np.random.randn() * ANGLE_NOISE * epsilon, -1, 1))

        reward = msg.get("reward", 0.0)

        # Guarda transición en el buffer
        if last_state is not None and last_action_data is not None:
            replay_buf.append({
                "state":         last_state,
                "next_state":    state,
                "reward":        reward,
                "done":          False,
                "action_shoot":  last_action_data["shoot"],
                "action_move_x": last_action_data["move_x"],
                "action_move_y": last_action_data["move_y"],
                "action_angle":  last_action_data["angle"],
            })

            # Entrenamiento en batch cuando hay suficientes muestras
            if len(replay_buf) >= BATCH_SIZE:
                indices = np.random.choice(len(replay_buf), BATCH_SIZE, replace=False)
                batch   = [replay_buf[i] for i in indices]
                nn.train_on_batch(batch)

        last_state       = state
        last_action_data = {
            "shoot":  action_shoot,
            "move_x": move_x,
            "move_y": move_y,
            "angle":  shot_angle,
        }

        resp = {
            "move_dir":  [move_x, move_y],
            "shot_angle": shot_angle,
            "action":    action_shoot,
        }
        sock.sendto(json.dumps(resp).encode(), addr)

    elif msg["type"] == "episode_end":
        reward = msg.get("reward", 0.0)

        if last_state is not None and last_action_data is not None:
            dummy = nn.forward(np.zeros(INPUTS))
            replay_buf.append({
                "state":         last_state,
                "next_state":    dummy,
                "reward":        reward,
                "done":          True,
                "action_shoot":  last_action_data["shoot"],
                "action_move_x": last_action_data["move_x"],
                "action_move_y": last_action_data["move_y"],
                "action_angle":  last_action_data["angle"],
            })

        nn.save()
        last_state       = None
        last_action_data = None

        print(
            f"[server] ep {msg.get('episode','?')} "
            f"| total_reward={msg.get('total_reward',0):.2f} "
            f"| final={reward:.2f} "
            f"| buf={len(replay_buf)}"
        )