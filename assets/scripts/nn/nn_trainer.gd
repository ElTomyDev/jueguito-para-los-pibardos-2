extends RefCounted
class_name NNTrainer

var lr: float = 0.0001          # Learning Rate (Inicia en 0.003 y baja a 0.001)
var gamma: float = 0.99        # Factor de descuento para recompensas futuras

# Regularización y estabilidad
const MAX_GRAD: float = 1.0
const WEIGHT_DECAY: float = 0.0001   # L2 decay (0.01% por actualización)
const WEIGHT_CLIP_MIN: float = -5.0   # Rango seguro para pesos
const WEIGHT_CLIP_MAX: float = 5.0

func train_step(nn: NeuralNetwork, state_act: Dictionary, next_state_act: Dictionary, reward: float, done: bool, action_taken: int) -> void:
	var v_s: float = state_act["critic_value"]
	var v_next: float = 0.0 if done else next_state_act["critic_value"]
	
	# Cálculo de la Ventaja (Temporal Difference Error)
	var td_target: float = reward + (gamma * v_next)
	var advantage: float = td_target - v_s
	advantage = clamp(advantage, -10.0, 10.0)
	
	# ---------------------------------------------
	# Backpropagation Manual de la cabeza del Critic
	# ---------------------------------------------
	var d_critic: float = -1.0 * advantage
	
		# Actualizar pesos Critic con weight decay y clipping
	for j in range(nn.hidden_size):
		var grad = lr * d_critic * state_act["hidden"][j]
		nn.W_critic[0][j] = nn.W_critic[0][j] * (1.0 - WEIGHT_DECAY) - grad
		nn.W_critic[0][j] = clamp(nn.W_critic[0][j], WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
	nn.b_critic[0] = nn.b_critic[0] * (1.0 - WEIGHT_DECAY) - lr * d_critic
	nn.b_critic[0] = clamp(nn.b_critic[0], WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
	
	# --------------------------------------------
	# Backpropagation Manual de la cabeza del Actor
	# --------------------------------------------
	var d_actor: Array = [0.0, 0.0, 0.0, 0.0]
	
	# Salidas continuas (move_x, move_y, shot_angle)
	for i in range(3):
		var tanh_grad: float = 1.0 - (state_act["actor_outputs"][i] * state_act["actor_outputs"][i])
		d_actor[i] = -advantage * tanh_grad
	
	# Acción discreta (disparar)
	var current_action_prob: float = state_act["actor_outputs"][3]
	var target_action: float = float(action_taken)
	d_actor[3] = -advantage * (target_action - current_action_prob)
	
	# Clip gradientes
	for i in range(d_actor.size()):
		d_actor[i] = clamp(d_actor[i], -MAX_GRAD, MAX_GRAD)
	d_critic = clamp(d_critic, -MAX_GRAD, MAX_GRAD)
	
		# Actualizar pesos Actor con weight decay y clipping
	for i in range(nn.actor_output_size):
		for j in range(nn.hidden_size):
			var grad = lr * d_actor[i] * state_act["hidden"][j]
			nn.W_actor[i][j] = nn.W_actor[i][j] * (1.0 - WEIGHT_DECAY) - grad
			nn.W_actor[i][j] = clamp(nn.W_actor[i][j], WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
		nn.b_actor[i] = nn.b_actor[i] * (1.0 - WEIGHT_DECAY) - lr * d_actor[i]
		nn.b_actor[i] = clamp(nn.b_actor[i], WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
		
	# ---------------------------------------------
	# Backpropagation hacia la Capa Oculta Compartida
	# ---------------------------------------------
	for i in range(nn.hidden_size):
		var grad_from_heads: float = 0.0
		for j in range(nn.actor_output_size):
			grad_from_heads += d_actor[j] * nn.W_actor[j][i]
		grad_from_heads += d_critic * nn.W_critic[0][i]
		
		var relu_grad: float = 1.0 if state_act["hidden"][i] > 0.0 else 0.0
		var d_h: float = grad_from_heads * relu_grad
		
		for j in range(nn.input_size):
			var grad = lr * d_h * state_act["inputs"][j]
			nn.W1[i][j] = nn.W1[i][j] * (1.0 - WEIGHT_DECAY) - grad
			nn.W1[i][j] = clamp(nn.W1[i][j], WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
		nn.b1[i] = nn.b1[i] * (1.0 - WEIGHT_DECAY) - lr * d_h
		nn.b1[i] = clamp(nn.b1[i], WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
