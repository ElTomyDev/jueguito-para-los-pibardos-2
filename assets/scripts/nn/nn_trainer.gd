extends RefCounted
class_name NNTrainer

var lr: float = 0.001          # Learning Rate (Inicia en 0.003 y baja a 0.001)
var gamma: float = 0.99        # Factor de descuento para recompensas futuras

const MAX_GRAD = 1.0

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
	
	# Actualizar pesos Critic
	for j in range(nn.hidden_size):
		nn.W_critic[0][j] -= lr * d_critic * state_act["hidden"][j]
	nn.b_critic[0] -= lr * d_critic
	
	# --------------------------------------------
	# Backpropagation Manual de la cabeza del Actor
	# --------------------------------------------
	var d_actor: Array = [0.0, 0.0, 0.0, 0.0]
	
	# Para salidas continuas (move_x, move_y, shot_angle)
	for i in range(3):
		var tanh_grad: float = 1.0 - (state_act["actor_outputs"][i] * state_act["actor_outputs"][i])
		# Gradiente para política determinística (DDPG-style)
		d_actor[i] = -advantage * tanh_grad
	
	# Para la acción discreta (disparar) - CORREGIDO
	var current_action_prob: float = state_act["actor_outputs"][3]  # salida sigmoide
	var target_action: float = float(action_taken)
	# Gradiente correcto: -advantage * (target - prob)
	d_actor[3] = -advantage * (target_action - current_action_prob)
	
	for i in range(d_actor.size()):
		d_actor[i] = clamp(d_actor[i], -MAX_GRAD, MAX_GRAD)
	d_critic = clamp(d_critic, -MAX_GRAD, MAX_GRAD)
	
	# Actualizar pesos Actor
	for i in range(nn.actor_output_size):
		for j in range(nn.hidden_size):
			nn.W_actor[i][j] -= lr * d_actor[i] * state_act["hidden"][j]
		nn.b_actor[i] -= lr * d_actor[i]
		
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
			nn.W1[i][j] -= lr * d_h * state_act["inputs"][j]
		nn.b1[i] -= lr * d_h

func _select_learning_rate() -> void:
	if GlobalVars.current_episode > 500 and lr > 0.001:
		lr = 0.001
