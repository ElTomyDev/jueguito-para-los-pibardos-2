extends RefCounted
class_name NeuralNetwork

# Arquitectura de la red
var input_size: int = 11
var hidden_size: int = 16
var actor_output_size: int = 4  # [move_x, move_y, shot_angle, action_logits]
var critic_output_size: int = 1 # [value]

# Pesos y Sesgos (Capa Oculta Compartida)
var W1: Array = [] # Matriz hidden_size x input_size
var b1: Array = [] # Vector hidden_size

# Pesos y Sesgos de la cabeza del Actor
var W_actor: Array = [] # Matriz actor_output_size x hidden_size
var b_actor: Array = [] # Vector actor_output_size

# Pesos y Sesgos de la cabeza del Critic
var W_critic: Array = [] # Matriz critic_output_size x hidden_size
var b_critic: Array = [] # Vector critic_output_size

func _init() -> void:
	_init_weights()

func _init_weights() -> void:
	# Inicialización de Xavier/Glorot para evitar desvanecimiento de gradiente
	W1 = _random_matrix(hidden_size, input_size, sqrt(2.0 / input_size))
	b1 = _zero_vector(hidden_size)
	
	W_actor = _random_matrix(actor_output_size, hidden_size, sqrt(2.0 / hidden_size))
	b_actor = _zero_vector(actor_output_size)
	
	W_critic = _random_matrix(critic_output_size, hidden_size, sqrt(2.0 / hidden_size))
	b_critic = _zero_vector(critic_output_size)

# --- Funciones de Activación ---
func _relu(x: float) -> float:
	return max(0.0, x)

func _tanh(x: float) -> float:
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

func _sigmoid(x: float) -> float:
	return 1.0 / (1.0 + exp(-x))

# --- Forward Pass ---
# Devuelve un Diccionario con las activaciones de todas las capas (necesario para el backpropagation)
func forward(inputs: Array) -> Dictionary:
	var h_act: Array = []
	for i in range(hidden_size):
		var sum: float = b1[i]
		for j in range(input_size):
			sum += inputs[j] * W1[i][j]
		h_act.append(_relu(sum))
	
	# Cabeza del Actor
	var actor_raw: Array = []
	for i in range(actor_output_size):
		var sum: float = b_actor[i]
		for j in range(hidden_size):
			sum += h_act[j] * W_actor[i][j]
		actor_raw.append(sum)
	
	# Mapeamos salidas del Actor: move_x, move_y y shot_angle usan Tanh (-1 a 1). Action usa Sigmoid para probabilidad.
	var actor_outputs: Array = [
		_tanh(actor_raw[0]),
		_tanh(actor_raw[1]),
		_tanh(actor_raw[2]),
		_sigmoid(actor_raw[3])
	]
	
	# Cabeza del Critic (Salida lineal para el valor del estado)
	var critic_val: float = b_critic[0]
	for j in range(hidden_size):
		critic_val += h_act[j] * W_critic[0][j]
	
	return {
		"inputs": inputs,
		"hidden": h_act,
		"actor_raw": actor_raw,
		"actor_outputs": actor_outputs,
		"critic_value": critic_val
	}

# --- Utilitarios Matemáticos ---
func _random_matrix(rows: int, cols: int, scale: float) -> Array:
	var mat: Array = []
	for i in range(rows):
		var row: Array = []
		for j in range(cols):
			row.append(randf_range(-1.0, 1.0) * scale)
		mat.append(row)
	return mat

func _zero_vector(size: int) -> Array:
	var vec: Array = []
	for i in range(size):
		vec.append(0.0)
	return vec
