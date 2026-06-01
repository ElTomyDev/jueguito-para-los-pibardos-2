extends Node
class_name NNTrainer

# ─────────────────────────────────────────
#  REINFORCE (Policy Gradient episódico)
# ─────────────────────────────────────────

const GAMMA         : float = 0.99   # Factor de descuento
const LEARNING_RATE : float = 0.001  # Tasa de aprendizaje
const EXPLORATION_SIGMA : float = 0.3  # Ruido de exploración base
const SIGMA_DECAY : float = 0.9995  # Multiplica por esto cada episodio
const SIGMA_MIN   : float = 0.05

var current_sigma : float = 0.5  # Empieza alto para explorar bien

# Referencia a la red
var nn : NeuralNetwork

# Buffers del episodio actual
var episode_inputs   : Array = [] 
var episode_outputs  : Array = []  # Salidas crudas (medias y probabilidades)
var episode_rewards  : Array = []
var episode_actions  : Array = []

# Métricas del episodio
var total_reward_last_episode : float = 0.0
var episode_count             : int   = 0

func setup(network: NeuralNetwork) -> void:
	nn = network

# ─────────────────────────────────────────
#  Llamar cada step durante el episodio
# ─────────────────────────────────────────
func record_step(input_vec: Array, raw_output: Array, action: Array, reward: float) -> void:
	episode_inputs.append(input_vec.duplicate())
	episode_outputs.append(raw_output.duplicate())
	episode_actions.append(action.duplicate())
	episode_rewards.append(reward)

# ─────────────────────────────────────────
#  Fin del episodio: calcula retornos y entrena la red
# ─────────────────────────────────────────
func end_episode() -> void:
	episode_count += 1
	if episode_rewards.is_empty():
		_clear_buffers()
		return
		
	# 1. Calcular los retornos con descuento (G_t)
	var returns : Array = []
	var g_t     : float = 0.0
	
	total_reward_last_episode = 0.0
	for r in episode_rewards:
		total_reward_last_episode += r
		
	# Calculamos hacia atrás: G_t = R_t + GAMMA * G_{t+1}
	for i in range(episode_rewards.size() - 1, -1, -1):
		g_t = episode_rewards[i] + GAMMA * g_t
		returns.insert(0, g_t)
		
	# 2. Normalizar retornos para estabilizar el gradiente (Media 0, Desviación Estándar 1)
	_normalize_returns(returns)
	
	# 3. Entrenar la red acumulando o aplicando el gradiente de política step por step
	for i in range(episode_inputs.size()):
		var input_vec  : Array = episode_inputs[i]
		var raw_output : Array = episode_outputs[i]
		var action     : Array = episode_actions[i]
		var return_val : float = returns[i]
		
		# Forzamos un forward pass para recargar los buffers internos de activación de la red
		var _current_out = nn.forward(input_vec)
		
		# Calculamos el gradiente de la política para este step
		var loss_gradient : Array = _compute_policy_gradient(raw_output, action, return_val)
		
		# Hacemos el backpropagation usando el gradiente calculado
		nn.backpropagate(loss_gradient, LEARNING_RATE)
		
	# 4. Decaer el factor de exploración (sigma)
	current_sigma = max(current_sigma * SIGMA_DECAY, SIGMA_MIN)
	
	# Limpiar buffers para el próximo episodio
	_clear_buffers()

func _clear_buffers() -> void:
	episode_inputs.clear()
	episode_outputs.clear()
	episode_rewards.clear()
	episode_actions.clear()

func _normalize_returns(returns: Array) -> void:
	if returns.size() <= 1:
		return
		
	var mean : float = 0.0
	for r in returns:
		mean += r
	mean /= returns.size()
	
	var variance : float = 0.0
	for r in returns:
		variance += (r - mean) * (r - mean)
	variance /= returns.size()
	
	var std : float = sqrt(variance) + 1e-8
	
	for i in range(returns.size()):
		returns[i] = (returns[i] - mean) / std

# ─────────────────────────────────────────
#  Cálculo matemático del Gradiente de Política para REINFORCE
# ─────────────────────────────────────────
func _compute_policy_gradient(raw_output: Array, action: Array, g_t: float) -> Array:
	var grad : Array = []
	var variance : float = current_sigma * current_sigma
	
	# --- OUTPUTS CONTINUOS (Índices 0, 1 y 2) ---
	# Salidas asociadas a la media de una distribución Gaussiana con activación Tanh.
	# Gradiente de la pérdida = -G_t * ((accion - media) / sigma^2)
	for i in range(3):
		var mean : float = raw_output[i]
		var act  : float = action[i]
		var d_log_pi : float = (act - mean) / variance
		
		# Guardamos el gradiente negativo para el descenso de gradiente de la red
		grad.append(-g_t * d_log_pi)
		
	# --- OUTPUT DISCRETO (Índice 3) ---
	# Salida asociada a una distribución de Bernoulli (disparar o no) vía Sigmoid.
	# Gradiente de log π = (accion - probabilidad)
	# Gradiente de la pérdida = -G_t * (accion - probabilidad)
	var prob : float = raw_output[3]
	var act_discreta : float = action[3]
	var d_log_pi_discrete : float = act_discreta - prob
	
	grad.append(-g_t * d_log_pi_discrete)
	
	return grad
