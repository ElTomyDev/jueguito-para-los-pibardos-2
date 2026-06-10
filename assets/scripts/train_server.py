import socket
import json
import numpy as np

HOST = "127.0.0.1"
PORT = 9999

INPUTS = 49
HIDDEN = 128
ACTOR_OUT = 4

GAMMA = 0.99

LR_ACTOR = 0.0003
LR_CRITIC = 0.001

ENTROPY_BETA = 0.01

MAX_GRAD = 1.0


# --------------------------------------------------
# UTILIDADES
# --------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def clip_grad(g):
    return np.clip(g, -MAX_GRAD, MAX_GRAD)


# --------------------------------------------------
# RED
# --------------------------------------------------

class ActorCritic:

    def __init__(self):

        s1 = np.sqrt(2.0 / INPUTS)
        s2 = np.sqrt(2.0 / HIDDEN)

        self.W1 = np.random.randn(HIDDEN, INPUTS) * s1
        self.b1 = np.zeros(HIDDEN)

        self.W_actor = np.random.randn(ACTOR_OUT, HIDDEN) * s2
        self.b_actor = np.zeros(ACTOR_OUT)

        self.W_critic = np.random.randn(1, HIDDEN) * s2
        self.b_critic = np.zeros(1)

    def forward(self, x):

        z1 = self.W1 @ x + self.b1

        h = np.maximum(0.0, z1)

        actor_raw = self.W_actor @ h + self.b_actor

        move_x = np.tanh(actor_raw[0])
        move_y = np.tanh(actor_raw[1])
        shot_angle = np.tanh(actor_raw[2])

        shoot_prob = sigmoid(actor_raw[3])

        value = float((self.W_critic @ h + self.b_critic)[0])

        return {
            "x": x,
            "h": h,
            "z1": z1,
            "actor_raw": actor_raw,
            "move_x": move_x,
            "move_y": move_y,
            "shot_angle": shot_angle,
            "shoot_prob": shoot_prob,
            "value": value
        }

    def train_step(
        self,
        state,
        next_state,
        reward,
        done,
        action
    ):

        value = state["value"]

        next_value = 0.0 if done else next_state["value"]

        td_target = reward + GAMMA * next_value

        advantage = td_target - value

        advantage = np.clip(advantage, -10.0, 10.0)

        h = state["h"]

        # ------------------------------------
        # CRITIC
        # ------------------------------------

        critic_error = value - td_target

        d_value = critic_error

        grad_wc = d_value * h
        grad_bc = d_value

        # ------------------------------------
        # ACTOR
        # ------------------------------------

        d_actor_raw = np.zeros(ACTOR_OUT)

        p = np.clip(
            state["shoot_prob"],
            1e-6,
            1.0 - 1e-6
        )

        # Policy Gradient Bernoulli

        d_actor_raw[3] = (
            -(action - p)
            * advantage
        )

        # Entropy Bonus

        entropy_grad = np.log(p / (1.0 - p))

        d_actor_raw[3] += (
            ENTROPY_BETA
            * entropy_grad
            * p
            * (1.0 - p)
        )

        # ------------------------------------
        # Backprop Actor
        # ------------------------------------

        grad_wa = np.outer(d_actor_raw, h)
        grad_ba = d_actor_raw

        # ------------------------------------
        # Hidden Layer
        # ------------------------------------

        dh_actor = self.W_actor.T @ d_actor_raw

        dh_critic = self.W_critic.T[:, 0] * d_value

        dh = dh_actor + dh_critic

        dh[state["z1"] <= 0.0] = 0.0

        grad_w1 = np.outer(dh, state["x"])
        grad_b1 = dh

        # ------------------------------------
        # Clipping
        # ------------------------------------

        grad_w1 = clip_grad(grad_w1)
        grad_b1 = clip_grad(grad_b1)

        grad_wa = clip_grad(grad_wa)
        grad_ba = clip_grad(grad_ba)

        grad_wc = clip_grad(grad_wc)
        grad_bc = clip_grad(grad_bc)

        # ------------------------------------
        # Update
        # ------------------------------------

        self.W_actor -= LR_ACTOR * grad_wa
        self.b_actor -= LR_ACTOR * grad_ba

        self.W_critic[0] -= LR_CRITIC * grad_wc
        self.b_critic[0] -= LR_CRITIC * grad_bc

        self.W1 -= LR_ACTOR * grad_w1
        self.b1 -= LR_ACTOR * grad_b1

    def save(self, path="./assets/train_data/boss_brain.json"):

        data = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W_actor": self.W_actor.tolist(),
            "b_actor": self.b_actor.tolist(),
            "W_critic": self.W_critic.tolist(),
            "b_critic": self.b_critic.tolist()
        }

        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path="./assets/train_data/boss_brain.json"):

        try:

            with open(path, "r") as f:
                d = json.load(f)

            self.W1 = np.array(d["W1"])
            self.b1 = np.array(d["b1"])

            self.W_actor = np.array(d["W_actor"])
            self.b_actor = np.array(d["b_actor"])

            self.W_critic = np.array(d["W_critic"])
            self.b_critic = np.array(d["b_critic"])

            print("[server] modelo cargado")

        except FileNotFoundError:

            print("[server] modelo nuevo")


# --------------------------------------------------
# SERVIDOR
# --------------------------------------------------

nn = ActorCritic()
nn.load()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"[server] escuchando {HOST}:{PORT}")

last_state = None
last_action = 0


while True:

    data, addr = sock.recvfrom(65535)

    msg = json.loads(data.decode())

    if msg["type"] == "step":
        x = np.array(
            msg["inputs"],
            dtype=np.float64
        )

        state = nn.forward(x)

        p_shoot = state["shoot_prob"]

        action = (
            1 if np.random.rand() < p_shoot
            else 0
        )

        move_dir = [
            float(state["move_x"]),
            float(state["move_y"])
        ]

        reward = msg.get(
            "reward",
            0.0
        )

        if last_state is not None:

            nn.train_step(
                last_state,
                state,
                reward,
                False,
                last_action
            )

        last_state = state
        last_action = action

        resp = {
            "move_dir": move_dir,
            "shot_angle": float(state["shot_angle"]),
            "action": action
        }

        sock.sendto(
            json.dumps(resp).encode(),
            addr
        )

    elif msg["type"] == "episode_end":

        reward = msg.get(
            "reward",
            0.0
        )

        if last_state is not None:

            dummy = nn.forward(
                np.zeros(INPUTS)
            )

            nn.train_step(
                last_state,
                dummy,
                reward,
                True,
                last_action
            )

        nn.save()

        last_state = None

        print(
            f"[server] episodio "
            f"{msg.get('episode', '?')} "
            f"| reward="
            f"{msg.get('total_reward', '?'):.2f}"
        )