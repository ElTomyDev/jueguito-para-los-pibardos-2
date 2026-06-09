import socket
import json
import numpy as np

HOST = "127.0.0.1"
PORT = 9999
INPUTS = 25
HIDDEN = 128
ACTOR_OUT = 4

# --- Red neuronal en NumPy ---
class ActorCritic:
    def __init__(self):
        scale1 = np.sqrt(2.0 / INPUTS)
        scale_a = np.sqrt(2.0 / HIDDEN)
        self.W1      = np.random.uniform(-scale1, scale1, (HIDDEN, INPUTS))
        self.b1      = np.zeros(HIDDEN)
        self.W_actor = np.random.uniform(-scale_a, scale_a, (ACTOR_OUT, HIDDEN))
        self.b_actor = np.zeros(ACTOR_OUT)
        self.W_critic= np.random.uniform(-scale_a, scale_a, (1, HIDDEN))
        self.b_critic= np.zeros(1)

    def forward(self, x: np.ndarray) -> dict:
        h = np.maximum(0, self.W1 @ x + self.b1)          # ReLU
        ar = self.W_actor @ h + self.b_actor
        ao = np.array([
            np.tanh(ar[0]), np.tanh(ar[1]),
            np.tanh(ar[2]), 1.0 / (1.0 + np.exp(-ar[3]))  # sigmoid
        ])
        v = float((self.W_critic @ h + self.b_critic)[0])
        return {"h": h, "ao": ao, "v": v, "x": x, "ar": ar}

    def train_step(self, state, next_state, reward, done, action):
        lr_a, lr_c, gamma = 0.00005, 0.0001, 0.99
        wd, max_g = 1e-6, 1.0

        v_s    = state["v"]
        v_next = 0.0 if done else next_state["v"]
        adv    = np.clip(reward + gamma * v_next - v_s, -2.0, 2.0)

        d_critic = np.clip(-adv, -max_g, max_g)

        d_actor = np.zeros(ACTOR_OUT)
        for i in range(3):
            av  = state["ao"][i]
            tg  = 1.0 - av * av
            d_actor[i] = np.clip(-adv * tg * (np.sign(av) if abs(av) > 0.01 else 1.0), -max_g, max_g)
        d_actor[3] = np.clip(-adv * (float(action) - state["ao"][3]), -max_g, max_g)

        # gradiente capa oculta
        dh = (d_actor @ self.W_actor + d_critic * self.W_critic[0])
        dh *= (state["h"] > 0).astype(float)

        # actualizar pesos
        self.W_critic[0] = self.W_critic[0] * (1 - wd) - lr_c * d_critic * state["h"]
        self.b_critic[0] = self.b_critic[0] * (1 - wd) - lr_c * d_critic

        self.W_actor = self.W_actor * (1 - wd) - lr_a * np.outer(d_actor, state["h"])
        self.b_actor = self.b_actor * (1 - wd) - lr_a * d_actor

        self.W1 = self.W1 * (1 - wd) - lr_a * np.outer(dh, state["x"])
        self.b1 = self.b1 * (1 - wd) - lr_a * dh

    def save(self, path="./assets/train_data/boss_brain.json"):
        data = {k: v.tolist() for k, v in {
            "W1": self.W1, "b1": self.b1,
            "W_actor": self.W_actor, "b_actor": self.b_actor,
            "W_critic": self.W_critic, "b_critic": self.b_critic
        }.items()}
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[server] modelo guardado en {path}")

    def load(self, path="./assets/train_data/boss_brain.json"):
        try:
            with open(path) as f:
                d = json.load(f)
            self.W1       = np.array(d["W1"])
            self.b1       = np.array(d["b1"])
            self.W_actor  = np.array(d["W_actor"])
            self.b_actor  = np.array(d["b_actor"])
            self.W_critic = np.array(d["W_critic"])
            self.b_critic = np.array(d["b_critic"])
            print(f"[server] modelo cargado desde {path}")
        except FileNotFoundError:
            print("[server] no hay modelo previo, usando pesos aleatorios")


# --- Replay buffer ---
class ReplayBuffer:
    def __init__(self, cap=512):
        self.buf, self.pos, self.cap = [], 0, cap

    def add(self, s, ns, r, done, a):
        t = (s, ns, r, done, a)
        if len(self.buf) < self.cap:
            self.buf.append(t)
        else:
            self.buf[self.pos] = t
        self.pos = (self.pos + 1) % self.cap

    def sample(self, n=16):
        idx = np.random.choice(len(self.buf), min(n, len(self.buf)), replace=False)
        return [self.buf[i] for i in idx]

    def ready(self, n=64):
        return len(self.buf) >= n


# --- Servidor UDP ---
nn  = ActorCritic()
nn.load()
buf = ReplayBuffer()
train_counter = 0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
print(f"[server] escuchando en {HOST}:{PORT}")

last_state = None
last_action = 0

while True:
    data, addr = sock.recvfrom(65535)
    msg = json.loads(data.decode())

    if msg["type"] == "step":
        inputs = np.array(msg["inputs"], dtype=np.float64)
        result = nn.forward(inputs)
        ao = result["ao"].tolist()

        # epsilon-greedy
        ep = msg.get("epsilon", 0.05)
        if np.random.rand() < ep:
            move_dir = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
            action   = np.random.randint(0, 2)
        else:
            move_dir = [ao[0], ao[1]]
            action   = 1 if ao[3] >= 0.2 else 0

        reward = msg.get("reward", 0.0)

        if last_state is not None:
            buf.add(last_state, result, reward, False, last_action)

        train_counter += 1
        if train_counter % 2 == 0 and buf.ready(64):
            for s, ns, r, d, a in buf.sample(16):
                nn.train_step(s, ns, r, d, a)

        last_state  = result
        last_action = action

        resp = {"move_dir": move_dir, "shot_angle": ao[2], "action": action}
        sock.sendto(json.dumps(resp).encode(), addr)

    elif msg["type"] == "episode_end":
        final_reward = msg.get("reward", 0.0)
        if last_state is not None:
            dummy = nn.forward(np.zeros(INPUTS))
            nn.train_step(last_state, dummy, final_reward, True, last_action)
        last_state = None
        nn.save()
        print(f"[server] episodio {msg.get('episode', '?')} | reward: {msg.get('total_reward', '?'):.2f}")