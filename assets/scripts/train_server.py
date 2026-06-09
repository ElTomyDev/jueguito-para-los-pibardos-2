import socket
import json
import numpy as np

HOST = "127.0.0.1"
PORT = 9999

INPUTS = 25
HIDDEN = 128
HIDDEN_ACTOR = 64
ACTOR_OUT = 4

GAMMA = 0.95

LR_SHARED = 0.0002  # Para W1 (capa compartida)
LR_ACTOR = 0.0002   # Para W2_actor y W_actor
LR_CRITIC = 0.0005  # Para W_critic

ENTROPY_BETA = 0.02
MAX_GRAD = 1.0

BEST_MODEL_PATH = "./assets/train_data/best_boss_brain.json"
TRAIN_DATA_PATH = "./assets/train_data/train_data.json"

BEST_WINDOW = 20
MIN_IMPROVEMENT = 5.0

best_avg_reward = -float("inf")
episode_rewards = []


def sigmoid(x) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def clip_grad(g):
    return np.clip(g, -MAX_GRAD, MAX_GRAD)

class ActorCritic:

    def __init__(self) -> None:
        s1 = np.sqrt(2.0 / INPUTS)
        s2 = np.sqrt(2.0 / HIDDEN)
        s3 = np.sqrt(2.0 / HIDDEN_ACTOR)

        # Capa compartida
        self.W1 = np.random.randn(HIDDEN, INPUTS) * s1
        self.b1 = np.zeros(HIDDEN)

        # Capa exclusiva del Actor
        self.W2_actor = np.random.randn(HIDDEN_ACTOR, HIDDEN) * s2
        self.b2_actor = np.zeros(HIDDEN_ACTOR)

        # Cabeza del Actor (opera sobre h2, no sobre h)
        self.W_actor = np.random.randn(ACTOR_OUT, HIDDEN_ACTOR) * s3
        self.b_actor = np.zeros(ACTOR_OUT)

        # Cabeza del Critic (opera directamente sobre h)
        self.W_critic = np.random.randn(1, HIDDEN) * s2
        self.b_critic = np.zeros(1)

    def forward(self, x) -> dict:
        # Capa compartida
        z1 = self.W1 @ x + self.b1
        h = np.maximum(0.0, z1)

        # Capa exclusiva del Actor
        z2_actor = self.W2_actor @ h + self.b2_actor
        h2 = np.maximum(0.0, z2_actor)

        # Outputs del Actor (desde h2)
        actor_raw = self.W_actor @ h2 + self.b_actor
        move_x     = np.tanh(actor_raw[0])
        move_y     = np.tanh(actor_raw[1])
        shot_angle = np.tanh(actor_raw[2])
        shoot_prob = sigmoid(actor_raw[3])

        # Output del Critic (desde h, sin pasar por W2_actor)
        value = float((self.W_critic @ h + self.b_critic)[0])

        return {
            "x":          x,
            "z1":         z1,
            "h":          h,
            "z2_actor":   z2_actor,
            "h2":         h2,
            "actor_raw":  actor_raw,
            "move_x":     move_x,
            "move_y":     move_y,
            "shot_angle": shot_angle,
            "shoot_prob": shoot_prob,
            "value":      value,
        }

    def train_step(self, state, next_state, reward, done, action) -> None:
        value      = state["value"]
        next_value = 0.0 if done else next_state["value"]
        td_target  = reward + GAMMA * next_value
        advantage  = np.clip(td_target - value, -10.0, 10.0)
        sigma = 0.5

        h  = state["h"]
        h2 = state["h2"]

        # --- Critic ---
        critic_error = value - td_target
        d_value = critic_error

        grad_wc = clip_grad(d_value * h)
        grad_bc = clip_grad(d_value)
        dh_from_critic = self.W_critic.T[:, 0] * d_value

        # --- Actor: calcular TODOS los gradientes primero ---
        d_actor_raw = np.zeros(ACTOR_OUT)

        # Continuos: gradiente gaussiano correcto (accion_tomada - media)
        d_actor_raw[0] = -advantage * (action['move_x']    - state["move_x"])    / (sigma**2)
        d_actor_raw[1] = -advantage * (action['move_y']    - state["move_y"])    / (sigma**2)
        d_actor_raw[2] = -advantage * (action['shot_angle']- state["shot_angle"])/ (sigma**2)

        # Discreto: policy gradient + entropy
        p = np.clip(state["shoot_prob"], 1e-6, 1.0 - 1e-6)
        d_actor_raw[3] = -(action['discrete'] - p) * advantage
        d_actor_raw[3] -= ENTROPY_BETA * np.log(p / (1.0 - p)) * p * (1.0 - p)

        # Gradientes de W_actor
        grad_wa = clip_grad(np.outer(d_actor_raw, h2))
        grad_ba = clip_grad(d_actor_raw)

        # Backprop a h2
        dh2 = self.W_actor.T @ d_actor_raw
        dh2[state["z2_actor"] <= 0.0] = 0.0

        grad_w2a = clip_grad(np.outer(dh2, h))
        grad_b2a = clip_grad(dh2)

        dh_from_actor = self.W2_actor.T @ dh2

        # Gradiente combinado hacia W1
        dh_combined = dh_from_critic + dh_from_actor
        dh_combined[state["z1"] <= 0.0] = 0.0

        grad_w1 = clip_grad(np.outer(dh_combined, state["x"]))
        grad_b1 = clip_grad(dh_combined)

        # Actualización
        self.W_critic[0] -= LR_CRITIC * grad_wc
        self.b_critic[0] -= LR_CRITIC * grad_bc
        self.W_actor     -= LR_ACTOR  * grad_wa
        self.b_actor     -= LR_ACTOR  * grad_ba
        self.W2_actor    -= LR_ACTOR  * grad_w2a
        self.b2_actor    -= LR_ACTOR  * grad_b2a
        self.W1          -= LR_SHARED * grad_w1
        self.b1          -= LR_SHARED * grad_b1

    def save(self, path="./assets/train_data/boss_brain.json") -> None:
        data = {
            "W1":       self.W1.tolist(),
            "b1":       self.b1.tolist(),
            "W2_actor": self.W2_actor.tolist(),
            "b2_actor": self.b2_actor.tolist(),
            "W_actor":  self.W_actor.tolist(),
            "b_actor":  self.b_actor.tolist(),
            "W_critic": self.W_critic.tolist(),
            "b_critic": self.b_critic.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path="./assets/train_data/boss_brain.json") -> None:
        try:
            with open(path, "r") as f:
                d = json.load(f)

            self.W1       = np.array(d["W1"])
            self.b1       = np.array(d["b1"])
            self.W2_actor = np.array(d["W2_actor"])
            self.b2_actor = np.array(d["b2_actor"])
            self.W_actor  = np.array(d["W_actor"])
            self.b_actor  = np.array(d["b_actor"])
            self.W_critic = np.array(d["W_critic"])
            self.b_critic = np.array(d["b_critic"])
            print("[server] modelo cargado")

        except FileNotFoundError:
            print("[server] modelo nuevo")
        except KeyError:
            # JSON viejo sin W2_actor: arranca con pesos nuevos para esa capa
            print("[server] JSON sin W2_actor, inicializando capa nueva")
            self.W1       = np.array(d["W1"])
            self.b1       = np.array(d["b1"])
            self.W_actor  = np.array(d["W_actor"])
            self.b_actor  = np.array(d["b_actor"])
            self.W_critic = np.array(d["W_critic"])
            self.b_critic = np.array(d["b_critic"])
            # W2_actor ya fue inicializado en __init__, no hace falta tocarlo

    def save_best(self, path=BEST_MODEL_PATH) -> None:
        data = {
            "W1":       self.W1.tolist(),
            "b1":       self.b1.tolist(),
            "W2_actor": self.W2_actor.tolist(),
            "b2_actor": self.b2_actor.tolist(),
            "W_actor":  self.W_actor.tolist(),
            "b_actor":  self.b_actor.tolist(),
            "W_critic": self.W_critic.tolist(),
            "b_critic": self.b_critic.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[BEST] modelo guardado -> {path}")


# --------------------------------------------------
# Stats y servidor (sin cambios lógicos)
# --------------------------------------------------

def load_train_data() -> None:
    global best_avg_reward, episode_rewards
    try:
        with open(TRAIN_DATA_PATH, "r") as f:
            data = json.load(f)
        best_avg_reward = data.get("best_avg_reward", -float("inf"))
        episode_rewards = data.get("episodes_rewards", [])
        print(f"[stats] cargadas {len(episode_rewards)} rewards")
    except FileNotFoundError:
        print("[stats] train_data.json nuevo")


def save_train_data() -> None:
    data = {
        "best_avg_reward": float(best_avg_reward),
        "episode_rewards": episode_rewards,
    }
    with open(TRAIN_DATA_PATH, "w") as f:
        json.dump(data, f, indent=4)


def update_best_model(nn, episode_reward) -> None:
    global best_avg_reward, episode_rewards
    episode_rewards.append(float(episode_reward))

    if len(episode_rewards) < BEST_WINDOW:
        save_train_data()
        return

    avg_reward = np.mean(episode_rewards[-BEST_WINDOW:])
    if avg_reward > (best_avg_reward + MIN_IMPROVEMENT):
        best_avg_reward = avg_reward
        nn.save_best()
        print(f"[BEST] nuevo promedio={avg_reward:.2f}")

    save_train_data()


nn = ActorCritic()
nn.load()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
print(f"[server] escuchando {HOST}:{PORT}")

last_state  = None
last_action = 0

while True:
    data, addr = sock.recvfrom(65535)
    msg = json.loads(data.decode())

    if msg["type"] == "step":
        x     = np.array(msg["inputs"], dtype=np.float64)
        state = nn.forward(x)

        p_shoot = state["shoot_prob"]
        epsilon = msg.get("epsilon", 0.1)
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 2)   # 0 o 1 aleatorio
        else:
            action = 1 if np.random.rand() < p_shoot else 0
        reward  = msg.get("reward", 0.0)

        if last_state is not None:
            nn.train_step(last_state, state, reward, False, last_action)

        # Generar acciones ruidosas (cliente las usará)
        noisy_move_x = np.clip(float(state["move_x"]) + np.random.normal(0, 0.2), -1.0, 1.0)
        noisy_move_y = np.clip(float(state["move_y"]) + np.random.normal(0, 0.2), -1.0, 1.0)
        noisy_angle = np.clip(float(state["shot_angle"]) + np.random.normal(0, 0.1), -1.0, 1.0)

        last_action = {
            "discrete": action,
            "move_x": noisy_move_x,
            "move_y": noisy_move_y,
            "shot_angle": noisy_angle,
        }

        resp = {
            "move_dir": [noisy_move_x, noisy_move_y],
            "shot_angle": noisy_angle,
            "action": action,
        }
        sock.sendto(json.dumps(resp).encode(), addr)

    elif msg["type"] == "episode_end":
        reward = msg.get("reward", 0.0)

        if last_state is not None:
            dummy = nn.forward(np.zeros(INPUTS))
            nn.train_step(last_state, dummy, reward, True, last_action)

        nn.save()

        total_reward = msg.get("total_reward", reward)
        update_best_model(nn, total_reward)
        last_state = None

        window_avg = (
            np.mean(episode_rewards[-BEST_WINDOW:])
            if len(episode_rewards) >= 1
            else 0.0
        )
        print(
            f"[server] episodio {msg.get('episode', '?')} "
            f"| reward={total_reward:.2f} "
            f"| avg20={window_avg:.2f} "
            f"| best={best_avg_reward:.2f}"
        )