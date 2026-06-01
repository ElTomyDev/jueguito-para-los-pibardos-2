extends Node
class_name NeuralNetwork

# ─────────────────────────────────────────
#  Arquitectura: 15 → 24 → 16 → 4
#  Activaciones: ReLU en capas ocultas
#                tanh  en outputs 0-2 (movimiento y ángulo)
#                sigmoid en output 3  (acción discreta de disparo)
# ─────────────────────────────────────────

const LAYER_SIZES : Array[int] = [15, 24, 16, 5]

const LEARNING_RATE : float = 0.001

# Pesos y biases por capa.
# weights[i] es una matriz [LAYER_SIZES[i+1]][LAYER_SIZES[i]]
# biases[i]  es un vector  [LAYER_SIZES[i+1]]
var weights : Array = []
var biases  : Array = []

# Guardamos las activaciones de cada capa para el backprop
var activations : Array = []   # activations[i] → vector post-activacion de capa i
var z_values    : Array = []   # z_values[i]    → vector pre-activacion  de capa i

func _init() -> void:
	_initialize_weights()

# ─────────────────────────────────────────
#  Inicialización de pesos (He initialization para ReLU)
# ─────────────────────────────────────────
func _initialize_weights() -> void:
	weights.clear()
	biases.clear()
	for i in range(LAYER_SIZES.size() - 1):
		var fan_in  : int = LAYER_SIZES[i]
		var fan_out : int = LAYER_SIZES[i + 1]
		var std     : float = sqrt(2.0 / fan_in)
		
		var w : Array = []
		for _j in range(fan_out):
			var row : Array = []
			for _k in range(fan_in):
				row.append(randf_range(-std, std))
			w.append(row)
		weights.append(w)
		
		var b : Array = []
		for _j in range(fan_out):
			b.append(0.0)
		biases.append(b)

# ─────────────────────────────────────────
#  Forward pass
# ─────────────────────────────────────────
func forward(input_vec: Array) -> Array:
	activations = [input_vec]
	z_values = []
	
	for i in range(weights.size()):
		var next_activations = []
		var next_z = []
		for j in range(LAYER_SIZES[i+1]):
			var z = biases[i][j]
			for k in range(LAYER_SIZES[i]):
				z += weights[i][j][k] * activations[i][k]
			next_z.append(z)
			
			# Activación:
			if i == weights.size() - 1:
				if j == 4: # Critic: Lineal
					next_activations.append(z)
				elif j == 3: # Disparo: Sigmoid
					next_activations.append(1.0 / (1.0 + exp(-clampf(z, -20.0, 20.0))))
				else: # Movimiento: Tanh
					next_activations.append(tanh(z))
			else:
				next_activations.append(max(0.0, z)) # ReLU
		z_values.append(next_z)
		activations.append(next_activations)
	return activations.back()

# ─────────────────────────────────────────
#  Backpropagation para REINFORCE
# ─────────────────────────────────────────
func backprop_actor_critic(grad_actor: Array, advantage: float) -> void:
	var num_layers = weights.size()
	var deltas = []
	deltas.resize(num_layers)

	# 1. Delta Capa Salida
	var last_delta = []
	for i in range(4): last_delta.append(grad_actor[i]) # Actor
	last_delta.append(-advantage) # Critic
	deltas[num_layers - 1] = last_delta

	# 2. Backprop Capas Ocultas
	for layer_idx in range(num_layers - 2, -1, -1):
		var current_delta = []
		var next_weights = weights[layer_idx + 1]
		var next_delta = deltas[layer_idx + 1]
		for j in range(LAYER_SIZES[layer_idx + 1]):
			var error = 0.0
			for k in range(next_delta.size()):
				error += next_weights[k][j] * next_delta[k]
			current_delta.append(error * (1.0 if z_values[layer_idx][j] > 0.0 else 0.0))
		deltas[layer_idx] = current_delta

	# 3. Actualización (LR diferenciada)
	var critic_lr = LEARNING_RATE * 0.5
	for i in range(num_layers):
		for j in range(deltas[i].size()):
			var lr = (critic_lr if (i == num_layers - 1 and j == 4) else LEARNING_RATE)
			biases[i][j] -= lr * deltas[i][j]
			for k in range(activations[i].size()):
				weights[i][j][k] -= lr * deltas[i][j] * activations[i][k]

# ─────────────────────────────────────────
#  Funciones de activación y sus derivadas (PROTEGIDAS contra NaN)
# ─────────────────────────────────────────
func _relu(x: float) -> float:
	return max(0.0, x)

func _relu_derivative(x: float) -> float:
	return 1.0 if x > 0.0 else 0.0

func _tanh(x: float) -> float:
	# Acotamos x para evitar desbordamientos en exp(x)
	var clamped_x : float = clampf(x, -20.0, 20.0)
	var e_pos : float = exp(clamped_x)
	var e_neg : float = exp(-clamped_x)
	return (e_pos - e_neg) / (e_pos + e_neg)

func _tanh_derivative(x: float) -> float:
	var t : float = _tanh(x)
	return 1.0 - t * t

func _sigmoid(x: float) -> float:
	# Clamping para evitar overflow con valores muy negativos
	var clamped_x : float = clampf(x, -20.0, 20.0)
	return 1.0 / (1.0 + exp(-clamped_x))

func _sigmoid_derivative(x: float) -> float:
	var s : float = _sigmoid(x)
	return s * (1.0 - s)
