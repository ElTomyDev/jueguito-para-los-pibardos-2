extends Node
class_name NeuralNetwork

# ─────────────────────────────────────────
#  Arquitectura: 15 → 24 → 16 → 4
#  Activaciones: ReLU en capas ocultas
#                tanh  en outputs 0-2 (movimiento y ángulo)
#                sigmoid en output 3  (acción discreta de disparo)
# ─────────────────────────────────────────

const LAYER_SIZES : Array[int] = [15, 24, 16, 4]

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
	if len(input_vec) != LAYER_SIZES[0]:
		push_error("La cantidad de inputs debe ser igual a lo que pide la red. Red: ", LAYER_SIZES[0], " Inputs: ", len(input_vec))
		return []
		
	activations.clear()
	z_values.clear()
	
	var current_activation : Array = input_vec.duplicate()
	activations.append(current_activation.duplicate())
	
	for layer_idx in range(weights.size()):
		var layer_weights : Array = weights[layer_idx]
		var layer_biases  : Array = biases[layer_idx]
		var next_activation : Array = []
		var layer_z         : Array = []
		
		# Multiplicación de matriz por vector + bias (Z = W * A + B)
		for j in range(layer_weights.size()):
			var neuron_weights : Array = layer_weights[j]
			var z : float = layer_biases[j]
			for k in range(current_activation.size()):
				z += neuron_weights[k] * current_activation[k]
			layer_z.append(z)
			
			# --- APLICACIÓN DE LAS FUNCIONES DE ACTIVACIÓN ---
			if layer_idx == weights.size() - 1:
				# Capa de salida (Capa 3, tamaño 4)
				if j < 3:
					# Índices 0, 1, 2: move_x, move_y, shot_angle -> Tanh
					next_activation.append(_tanh(z))
				else:
					# Índice 3: current_action -> Sigmoid
					next_activation.append(_sigmoid(z))
			else:
				# Capas ocultas -> ReLU
				next_activation.append(_relu(z))
				
		z_values.append(layer_z)
		current_activation = next_activation.duplicate()
		activations.append(current_activation.duplicate())
		
	return current_activation

# ─────────────────────────────────────────
#  Backpropagation para REINFORCE
# ─────────────────────────────────────────
func backpropagate(loss_gradient: Array, learning_rate: float) -> void:
	var deltas : Array = []
	for i in range(weights.size()):
		deltas.append([])
		
	# 1. Calcular el delta para la capa de salida (Última capa)
	var output_layer_idx : int = weights.size() - 1
	var output_z         : Array = z_values[output_layer_idx]
	var output_delta     : Array = []
	
	for j in range(loss_gradient.size()):
		var d_activation : float = 0.0
		if j < 3:
			# Índices 0, 1, 2 usan la derivada de la Tanh
			d_activation = _tanh_derivative(output_z[j])
		else:
			# Índice 3 usa la derivada de la Sigmoid
			d_activation = _sigmoid_derivative(output_z[j])
			
		output_delta.append(loss_gradient[j] * d_activation)
		
	deltas[output_layer_idx] = output_delta
	
	# 2. Propagar el error hacia atrás (Capas ocultas)
	for layer_idx in range(output_layer_idx - 1, -1, -1):
		var layer_weights_next : Array = weights[layer_idx + 1]
		var delta_next         : Array = deltas[layer_idx + 1]
		var current_z          : Array = z_values[layer_idx]
		var current_delta      : Array = []
		
		# Corregido: Iteramos según la cantidad de neuronas de la capa actual
		for j in range(current_z.size()):
			var error : float = 0.0
			for k in range(delta_next.size()):
				error += layer_weights_next[k][j] * delta_next[k]
			current_delta.append(error * _relu_derivative(current_z[j]))
		deltas[layer_idx] = current_delta
		
	# 3. Actualización de pesos y biases utilizando Descenso de Gradiente
	for layer_idx in range(weights.size()):
		var delta      : Array = deltas[layer_idx]
		var act_prev   : Array = activations[layer_idx]
		
		for j in range(delta.size()):
			biases[layer_idx][j] -= learning_rate * delta[j]
			for k in range(act_prev.size()):
				weights[layer_idx][j][k] -= learning_rate * delta[j] * act_prev[k]

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
