import numpy as np
import json
import threading
import socket

# Hiperparámetros PPO
INPUTS      = 40
HIDDEN      = 256
ACTOR_OUT   = 4
GAMMA       = 0.99
LAMBDA      = 0.95      # GAE lambda
LR          = 0.0001
CLIP_EPS    = 0.2       # clip de PPO
ENTROPY_B   = 0.01
MAX_GRAD    = 0.5
CHUNK_LEN  = 64    # largo de cada chunk contiguo
EPOCHS     = 4     # pasadas sobre el episodio por update

is_updating = False

def sigmoid(x) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def clip_grad(g) -> any:
    return np.clip(g, -MAX_GRAD, MAX_GRAD)

def ppo_eff_ratio(ratio, clipped_ratio, adv) -> float:
    loss_unclipped = ratio * adv
    loss_clipped   = clipped_ratio * adv
    # PPO toma el mínimo: si el término sin clip ya es menor, se usa ratio.
    # Si el clipeado es menor (clip activo), gradiente = 0.
    return ratio if loss_unclipped <= loss_clipped else 0.0


class LSTMCell:
    """LSTM minimal implementado en numpy"""
    def __init__(self, input_size, hidden_size) -> None:
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

        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, h_prev, c_prev) -> any:
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
        c_new = np.clip(c_new, -10.0, 10.0)
        h_new = o * np.tanh(c_new)

        cache = {
            "x": x, "h_prev": h_prev, "c_prev": c_prev,
            "xh": xh, "f": f, "i": i,
            "c_tilde": c_tilde, "o": o,
            "c_new": c_new, "h_new": h_new
        }
        return h_new, c_new, cache

    def zero_state(self) -> any:
        return np.zeros(self.hidden_size), np.zeros(self.hidden_size)

    def backward(self, dh_next, dc_next, cache) -> any:
        """
        dh_next: gradiente respecto a h_new (desde arriba)
        dc_next: gradiente respecto a c_new (desde arriba)
        cache:  diccionario guardado en forward()
        Retorna: dx, dh_prev, dc_prev, gradientes de los pesos
        """
        xh = cache["xh"]
        f = cache["f"]
        i = cache["i"]
        o = cache["o"]
        c_tilde = cache["c_tilde"]
        c_prev = cache["c_prev"]
        c_new = cache["c_new"]

        # Gradiente de la salida
        do = dh_next * np.tanh(c_new)
        do = do * o * (1 - o) # sigmoid derivative
        
        # Gradiente de c_new que viene de h_new a través del tanh
        dc_from_o = dh_next * o * (1 - np.tanh(c_new)**2)

        # Gradiente de la celda
        dc = dc_next + dc_from_o
        dc = np.clip(dc, -10.0, 10.0)

        # Gradientes de las compuertas
        df = dc * c_prev * f * (1 - f)       # forget gate
        di = dc * c_tilde * i * (1 - i)      # input gate
        dc_tilde = dc * i * (1 - c_tilde**2) # candidate
        do_input = do                         # output gate

        # Acumular gradientes para pesos
        dWf = np.outer(df, xh)
        dbf = df
        dWi = np.outer(di, xh)
        dbi = di
        dWc = np.outer(dc_tilde, xh)
        dbc = dc_tilde
        dWo = np.outer(do_input, xh)
        dbo = do_input

        # Gradiente para la entrada xh (concat de x y h_prev)
        dxh = (self.Wf.T @ df) + (self.Wi.T @ di) + (self.Wc.T @ dc_tilde) + (self.Wo.T @ do_input)
        dx = dxh[:self.input_size]      # primeros input_size elementos
        dh_prev = dxh[self.input_size:] # los siguientes hidden_size
        dc_prev = dc * f

        return dx, dh_prev, dc_prev, (dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo)

class PPOActorCritic:
    def __init__(self) -> None:
        self.lock = threading.Lock()
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
        self.log_std = np.clip(self.log_std, -5.0, 2.0)

    def forward(self, x, h, c) -> dict:
        with self.lock:
            W_in    = self.W_in.copy()
            b_in    = self.b_in.copy()
            W_actor = self.W_actor.copy()
            b_actor = self.b_actor.copy()
            W_critic= self.W_critic.copy()
            b_critic= self.b_critic.copy()
            log_std = self.log_std.copy()

        z = np.maximum(0.0, W_in @ x + b_in)  # ReLU
        h_new, c_new, cache = self.lstm.forward(z, h, c)

        actor_raw  = W_actor  @ h_new + b_actor
        value      = float((W_critic @ h_new + b_critic)[0])
        std        = np.exp(log_std)
        means      = np.tanh(actor_raw[:3])
        shoot_prob = sigmoid(actor_raw[3])

        return {
            "h": h_new,
            "c": c_new,
            "z": z,
            "x": x,
            "lstm_cache": cache,
            "actor_raw": actor_raw,
            "means": means,
            "std": std,
            "shoot_prob": shoot_prob,
            "value": value,
            "_W_actor": W_actor,
            "_b_actor": b_actor,
            "_W_critic": W_critic,
            "_b_critic": b_critic,
            "_W_in": W_in,
            "_b_in": b_in,

        }

    def sample_action(self, state, epsilon=0.1) -> any:
        """Muestrea acción con exploración gaussiana"""
        noise  = np.random.randn(3) * state["std"] * epsilon
        move_x = float(np.clip(state["means"][0] + noise[0], -1, 1))
        move_y = float(np.clip(state["means"][1] + noise[1], -1, 1))
        angle  = float(np.clip(state["means"][2] + noise[2], -1, 1))
        shoot  = 1 if np.random.rand() < state["shoot_prob"] else 0
        return move_x, move_y, angle, shoot

    def log_prob(self, state, actions) -> any:
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

    def compute_gae(self, rewards, values, dones) -> any:
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

    def bptt_update(self, trajectories) -> None:
        """
        trajectories: lista de diccionarios en orden temporal, cada uno con:
            'state', 'actions', 'old_log_prob', 'advantage', 'return_'
        """
        # Inicializar gradientes acumulados
        gW_in = np.zeros_like(self.W_in)
        gb_in = np.zeros_like(self.b_in)
        gWa   = np.zeros_like(self.W_actor)
        gba   = np.zeros_like(self.b_actor)
        gWc   = np.zeros_like(self.W_critic)
        gbc   = np.zeros_like(self.b_critic)

        # Gradientes del LSTM (para el episodio completo)
        g_lstm_Wf = np.zeros_like(self.lstm.Wf)
        g_lstm_bf = np.zeros_like(self.lstm.bf)
        g_lstm_Wi = np.zeros_like(self.lstm.Wi)
        g_lstm_bi = np.zeros_like(self.lstm.bi)
        g_lstm_Wc = np.zeros_like(self.lstm.Wc)
        g_lstm_bc = np.zeros_like(self.lstm.bc)
        g_lstm_Wo = np.zeros_like(self.lstm.Wo)
        g_lstm_bo = np.zeros_like(self.lstm.bo)

        # Estado oculto inicial y gradientes iniciales (cero al final del episodio)
        dh_next = np.zeros(self.lstm.hidden_size)
        dc_next = np.zeros(self.lstm.hidden_size)

        # Recorrer el episodio en orden inverso
        for tr in reversed(trajectories):
            state = tr["state"]
            adv = tr["advantage"]
            ret = tr["return_"]
            old_lp = tr["old_log_prob"]
            actions = tr["actions"]

            # ----- Pérdida PPO (igual que antes) -----
            new_lp = self.log_prob(state, actions)
            ratio = np.exp(np.clip(new_lp - old_lp, -10, 10))
            clipped_ratio = np.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            v_pred = state["value"]
            p = np.clip(state["shoot_prob"], 1e-6, 1 - 1e-6)
            entropy = -(p * np.log(p) + (1-p) * np.log(1-p))
            entropy += 0.5 * np.sum(np.log(2 * np.pi * np.e * state["std"] ** 2))

            # Gradientes para actor y critic (igual que antes)
            d_raw = np.zeros(ACTOR_OUT)
            std = state["std"]

            # Para outputs continuos:
            for idx in range(3):
                a_exec = float(np.clip(actions[idx], -0.999, 0.999))
                mu = state["means"][idx]
                eff = ppo_eff_ratio(ratio, clipped_ratio, adv)
                d_raw[idx] = -eff * adv * (a_exec - mu) / (std[idx] ** 2 + 1e-8)

            
            # Para la acción discreta:
            eff = ppo_eff_ratio(ratio, clipped_ratio, adv)
            d_raw[3] = -eff * adv * (actions[3] - p)
            d_raw[3] += ENTROPY_B * np.log(p / (1 - p)) * p * (1 - p)

            # Gradientes actor/critic
            h = state["h"]
            gWa += np.outer(d_raw, h)
            gba += d_raw
            d_val = v_pred - ret
            gWc += np.outer(d_val, h)
            gbc += d_val

            # Gradiente que viene de las cabezas hacia la salida del LSTM
            dh = self.W_actor.T @ d_raw + self.W_critic.T[:, 0] * d_val
            # Añadir el dh_next acumulado del paso siguiente (BPTT)
            dh_total = dh + dh_next

            # Obtener el cache de este paso
            cache = state["lstm_cache"]

            dh_total = np.clip(dh_total, -5.0, 5.0)
            dc_next = np.clip(dc_next, -5.0, 5.0)
            
            # Llamar al backward de la celda LSTM
            dx, dh_prev, dc_prev, lstm_grads = self.lstm.backward(dh_total, dc_next, cache)
            (dWf_step, dbf_step, dWi_step, dbi_step,
            dWc_step, dbc_step, dWo_step, dbo_step) = lstm_grads

            # Acumular gradientes del LSTM
            g_lstm_Wf += dWf_step
            g_lstm_bf += dbf_step
            g_lstm_Wi += dWi_step
            g_lstm_bi += dbi_step
            g_lstm_Wc += dWc_step
            g_lstm_bc += dbc_step
            g_lstm_Wo += dWo_step
            g_lstm_bo += dbo_step

            # Gradientes para la capa de entrada (ReLU)
            # dx ya es el gradiente respecto a z (salida de la capa anterior)
            #dz = dx * (state["z"] > 0).astype(float)   # derivada ReLU
            dz = dx * (state["z"] > 0).astype(float) + 1e-8 * dx   # evita ceros exactos
            x_input = state["x"]  # es el input original de este paso
            gW_in += np.outer(dz, x_input)
            gb_in += dz

            # Actualizar dh_next y dc_next para el paso anterior
            dh_next = np.clip(dh_prev, -5.0, 5.0)
            dc_next = np.clip(dc_prev, -5.0, 5.0)

        # Una vez recorrido todo el episodio, aplicar la actualización de pesos
        n = len(trajectories)
        with self.lock:
            self.W_actor   -= LR * clip_grad(gWa / n)
            self.b_actor   -= LR * clip_grad(gba / n)
            self.W_critic -= LR * clip_grad(gWc / n)
            self.b_critic -= LR * clip_grad(gbc / n)
            self.W_in       -= LR * clip_grad(gW_in / n)
            self.b_in       -= LR * clip_grad(gb_in / n)
            self.lstm.Wf    -= LR * clip_grad(g_lstm_Wf / n)
            self.lstm.bf    -= LR * clip_grad(g_lstm_bf / n)
            self.lstm.Wi    -= LR * clip_grad(g_lstm_Wi / n)
            self.lstm.bi    -= LR * clip_grad(g_lstm_bi / n)
            self.lstm.Wc    -= LR * clip_grad(g_lstm_Wc / n)
            self.lstm.bc    -= LR * clip_grad(g_lstm_bc / n)
            self.lstm.Wo    -= LR * clip_grad(g_lstm_Wo / n)
            self.lstm.bo    -= LR * clip_grad(g_lstm_bo / n)
    
    def ppo_chunk_update(self, trajectories) -> None:
        """
        Divide el episodio en chunks contiguos de CHUNK_LEN steps.
        Por cada época, recorre los chunks en orden y propaga h/c
        entre ellos para mantener el contexto del LSTM intacto.
        
        Ventajas sobre bptt_update() en episodios de 900 steps:
        - El gradiente retrocede solo CHUNK_LEN pasos → no se desvanece
        - EPOCHS pasadas → más updates por episodio (EPOCHS * n_chunks)
        - h/c se propagan hacia adelante → el LSTM retiene contexto
        """
        n = len(trajectories)
        if n < 2:
            return

        # Arma lista de chunks: índices [start, end)
        chunks = []
        for start in range(0, n, CHUNK_LEN):
            end = min(start + CHUNK_LEN, n)
            chunks.append((start, end))

        for epoch in range(EPOCHS):
            # Reinicia el estado oculto al comienzo de cada época
            h_chunk = np.zeros(self.lstm.hidden_size)
            c_chunk = np.zeros(self.lstm.hidden_size)

            for (start, end) in chunks:
                chunk = trajectories[start:end]
                h_chunk, c_chunk = self._update_chunk(
                    chunk, h_chunk, c_chunk
                )

    def _update_chunk(self, chunk, h_init, c_init) -> any:
        """
        Hace un paso de BPTT sobre un chunk contiguo.
        
        - Forward pass: recorre el chunk propagando h/c desde h_init/c_init
        - Backward pass: BPTT solo sobre el chunk (no sobre el episodio entero)
        - Retorna el h/c final para pasarle al próximo chunk
        
        Nota: re-ejecuta el forward con h_init/c_init en lugar de usar
        los caches guardados, porque esos caches corresponden al estado
        del episodio original (sin el h correcto para este chunk).
        """
        # Inicializar gradientes acumulados
        gW_in = np.zeros_like(self.W_in)
        gb_in = np.zeros_like(self.b_in)
        gWa   = np.zeros_like(self.W_actor)
        gba   = np.zeros_like(self.b_actor)
        gWc   = np.zeros_like(self.W_critic)
        gbc = np.zeros(1) 
        g_lstm = {k: np.zeros_like(getattr(self.lstm, k))
                for k in ("Wf","bf","Wi","bi","Wc","bc","Wo","bo")}

        # ---- Forward pass del chunk ----
        # Re-ejecuta con h_init/c_init para tener los caches correctos
        h, c = h_init.copy(), c_init.copy()
        fresh_states = []
        for tr in chunk:
            x = tr["state"]["x"]
            z = np.maximum(0.0, self.W_in @ x + self.b_in)
            h, c, lstm_cache = self.lstm.forward(z, h, c)

            actor_raw  = self.W_actor  @ h + self.b_actor
            value      = float((self.W_critic @ h + self.b_critic)[0])
            std        = np.exp(self.log_std)
            means      = np.tanh(actor_raw[:3])
            shoot_prob = sigmoid(actor_raw[3])

            fresh_states.append({
                "x": x, "z": z, "h": h, "c": c,
                "lstm_cache": lstm_cache,
                "actor_raw": actor_raw,
                "means": means, "std": std,
                "shoot_prob": shoot_prob,
                "value": value,
            })

        h_final, c_final = h.copy(), c.copy()

        # ---- Backward pass (BPTT truncado al chunk) ----
        dh_next = np.zeros(self.lstm.hidden_size)
        dc_next = np.zeros(self.lstm.hidden_size)

        for i, (tr, fs) in enumerate(
                zip(reversed(chunk), reversed(fresh_states))):
            adv    = tr["advantage"]
            ret    = tr["return_"]
            old_lp = tr["old_log_prob"]
            actions = tr["actions"]

            # PPO ratio con log_prob recalculado sobre el estado fresco
            new_lp = self.log_prob(fs, actions)
            ratio  = np.exp(np.clip(new_lp - old_lp, -10, 10))
            clipped_ratio = np.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)

            # Gradiente actor (continuo + discreto)
            d_raw = np.zeros(ACTOR_OUT)
            std   = fs["std"]
            for idx in range(3):
                mu = fs["means"][idx]
                a  = float(np.clip(actions[idx], -0.999, 0.999))
                eff = ppo_eff_ratio(ratio, clipped_ratio, adv)
                d_raw[idx] = -eff * adv * (a - mu) / (std[idx]**2 + 1e-8)
            
            p = np.clip(fs["shoot_prob"], 1e-6, 1 - 1e-6)
            eff = ppo_eff_ratio(ratio, clipped_ratio, adv)
            d_raw[3]  = -eff * adv * (actions[3] - p)
            d_raw[3] += ENTROPY_B * np.log(p / (1 - p)) * p * (1 - p)

            # Gradiente critic
            d_val = fs["value"] - ret

            # Acumular cabezas
            gWa += np.outer(d_raw, fs["h"])
            gba += d_raw
            gWc += np.outer(np.array([d_val]), fs["h"])
            gbc += np.array([d_val])

            # Gradiente hacia h desde las cabezas + BPTT acumulado
            dh = self.W_actor.T @ d_raw + self.W_critic.T[:, 0] * d_val
            dh_total = np.clip(dh + dh_next, -5.0, 5.0)
            dc_next  = np.clip(dc_next, -5.0, 5.0)

            dx, dh_prev, dc_prev, lstm_grads = self.lstm.backward(
                dh_total, dc_next, fs["lstm_cache"]
            )
            (dWf, dbf, dWi, dbi, dWc_l, dbc, dWo, dbo) = lstm_grads

            g_lstm["Wf"] += dWf;  g_lstm["bf"] += dbf
            g_lstm["Wi"] += dWi;  g_lstm["bi"] += dbi
            g_lstm["Wc"] += dWc_l; g_lstm["bc"] += dbc
            g_lstm["Wo"] += dWo;  g_lstm["bo"] += dbo

            dz     = dx * (fs["z"] > 0).astype(float)
            gW_in += np.outer(dz, fs["x"])
            gb_in += dz

            dh_next = dh_prev
            dc_next = dc_prev

        # ---- Aplicar gradientes ----
        m = len(chunk)
        with self.lock:
            self.W_actor  -= LR * clip_grad(gWa / m)
            self.b_actor  -= LR * clip_grad(gba / m)
            self.W_critic -= LR * clip_grad(gWc / m)
            self.b_critic -= LR * clip_grad(gbc / m)
            self.W_in     -= LR * clip_grad(gW_in / m)
            self.b_in     -= LR * clip_grad(gb_in / m)
            for k in g_lstm:
                getattr(self.lstm, k)[:] -= LR * clip_grad(g_lstm[k] / m)

        return h_final, c_final
        
    def save(self, path="./assets/train_data/boss_brain.json") -> None:
        with self.lock:
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

    def load(self, path="./assets/train_data/boss_brain.json") -> None:
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
        traj_len = 0
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
            traj_len = len(trajectories)
            #nn.bptt_update(trajectories)
            #threading.Thread(target=nn.save, daemon=True).start()
            def update_and_save(traj) -> None:
                nn.bptt_update(traj)
                #nn.ppo_chunk_update(traj)
                nn.save()
                
            threading.Thread(target=update_and_save, args=(trajectories,), daemon=True).start()
        else:
            nn.save()

        # Reset para el próximo episodio
        episode_buffer.clear()
        last_state   = None
        last_actions = None
        h_state, c_state = nn.lstm.zero_state()

        
        print(
            f"[server] ep {msg.get('episode','?')} "
            f"| total={msg.get('total_reward', 0):.2f} "
            f"| final={final_reward:.2f} "
            f"| traj={traj_len}"
        )