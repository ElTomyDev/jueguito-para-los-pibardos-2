extends Node
class_name NNTrainer

# ─────────────────────────────────────────
#  REINFORCE (Policy Gradient episódico)
#
#  Cada step del episodio guardamos:
#    - los outputs de la red (la "política")
#    - la recompensa recibida ese step
#
#  Al fin del episodio calculamos el retorno
#  acumulado con descuento (G_t) y usamos eso
#  para escalar el gradiente y actualizar la red.
# ─────────────────────────────────────────

const GAMMA         : float = 0.99   # Factor de descuento
const LEARNING_RATE : float = 0.001  # Tasa de aprendizaje

# Referencia a la red
var nn : NeuralNetwork

# Buffers del episodio actual
var episode_inputs   : Array = []  # Array de Array[float]
var episode_outputs  : Array = []  # Array de Array[float] (outputs crudos de la red)
var episode_rewards  : Array = []  # Array de float

# Métricas del episodio (útil para debug / UI)
var total_reward_last_episode : float = 0.0
var episode_count             : int   = 0

func setup(network: NeuralNetwork) -> void:
	nn = network

# ─────────────────────────────────────────
#  Llamar cada step durante el episodio
#  input_vec:   inputs normalizados del boss
#  reward:      recompensa de este step
#  Retorna:     output de la red (move_dir, shot_dir, action)
# ─────────────────────────────────────────
func step(input_vec: Array, reward: float) -> Array:
	var output : Array = nn.forward(input_vec)
	
	episode_inputs.append(input_vec.duplicate())
	episode_outputs.append(output.duplicate())
	episode_rewards.append(reward)
	
	return output

# ─────────────────────────────────────────
#  Llamar al final del episodio (cuando muere el boss o el jugador)
#  final_reward: recompensa terminal (grande positiva/negativa)
# ─────────────────────────────────────────
func end_episode(final_reward: float) -> void:
	if episode_rewards.is_empty():
		return
	
	episode_rewards[episode_rewards.size() - 1] += final_reward
	total_reward_last_episode = 0.0
	for r in episode_rewards:
		total_reward_last_episode += r
	
	_update_network()
	_clear_buffers()
	episode_count += 1

# ─────────────────────────────────────────
#  Calcula retornos con descuento G_t y
#  hace backprop para cada step del episodio
# ─────────────────────────────────────────
func _update_network() -> void:
	var returns : Array = _compute_returns()
	_normalize_returns(returns)
	
	for t in range(episode_inputs.size()):
		# Re-ejecutamos el forward para restaurar activaciones en la red
		nn.forward(episode_inputs[t])
		
		# Gradiente de política: -G_t * d_log_pi/d_theta
		# Para outputs continuos (tanh): gradiente = -G_t * (target - output)
		# Usamos el output actual como "target" desplazado por el retorno
		var grad : Array = _compute_policy_gradient(episode_outputs[t], returns[t])
		nn.backward(grad, LEARNING_RATE)

# ─────────────────────────────────────────
#  Retorno acumulado con descuento desde cada step t
#  G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
# ─────────────────────────────────────────
func _compute_returns() -> Array:
	var returns : Array = []
	returns.resize(episode_rewards.size())
	var running_return : float = 0.0
	
	for t in range(episode_rewards.size() - 1, -1, -1):
		running_return = episode_rewards[t] + GAMMA * running_return
		returns[t] = running_return
	
	return returns

# ─────────────────────────────────────────
#  Normaliza los retornos (media 0, std 1)
#  para estabilizar el entrenamiento
# ─────────────────────────────────────────
func _normalize_returns(returns: Array) -> void:
	if returns.size() < 2:
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
#  Gradiente de política para REINFORCE
#  Para cada output, el gradiente es:
#    -G_t * (output_actual - lo_que_deberia_haber_hecho)
#
#  Para outputs continuos (move_dir, shot_dir):
#    "lo_que_deberia_haber_hecho" = el output desplazado
#    en la dirección del retorno. Si G_t > 0, reforzamos
#    lo que hizo. Si G_t < 0, lo penalizamos.
#
#  Para el output discreto (current_action via sigmoid):
#    Usamos el gradiente de log-probabilidad de Bernoulli.
# ─────────────────────────────────────────
func _compute_policy_gradient(outputs: Array, g_t: float) -> Array:
	var grad : Array = []
	
	# Outputs 0-3: direcciones continuas (tanh)
	for i in range(4):
		# Gradiente = -G_t * output (política gaussiana simplificada)
		# Si G_t > 0: refuerza la acción tomada
		# Si G_t < 0: penaliza la acción tomada
		grad.append(-g_t * outputs[i])
	
	# Output 4: acción discreta (sigmoid → Bernoulli)
	# grad = -(G_t * (1 - p)) si acción=1, -(G_t * (-p)) si acción=0
	var p      : float = outputs[4]          # probabilidad de acción=1
	var action : int   = 1 if p >= 0.5 else 0
	var bernoulli_grad : float
	if action == 1:
		bernoulli_grad = -g_t * (1.0 - p)
	else:
		bernoulli_grad = g_t * p
	grad.append(bernoulli_grad)
	
	return grad

func _clear_buffers() -> void:
	episode_inputs.clear()
	episode_outputs.clear()
	episode_rewards.clear()
