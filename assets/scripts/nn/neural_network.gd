extends Node
class_name NeuralNetwork

# ─────────────────────────────────────────
#  Arquitectura: 12 → 24 → 16 → 5
#  Activaciones: ReLU en capas ocultas
#                tanh  en outputs 0-3 (direcciones continuas)
#                sigmoid en output 4  (accion discreta)
# ─────────────────────────────────────────

const LAYER_SIZES : Array[int] = [14, 24, 16, 5]

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
#  Inicializacion de pesos (He initialization para ReLU)
# ─────────────────────────────────────────
func _initialize_weights() -> void:
	weights.clear()
	biases.clear()
	for i in range(LAYER_SIZES.size() - 1):
		var fan_in  : int = LAYER_SIZES[i]
		var fan_out : int = LAYER_SIZES[i + 1]
		var std     : float = sqrt(2.0 / fan_in)  # He init
		
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
#  input_vec: Array[float] de tamaño LAYER_SIZES[0]
#  Retorna:   Array[float] de tamaño LAYER_SIZES[-1]
# ─────────────────────────────────────────
func forward(input_vec: Array) -> Array:
	if len(input_vec) != LAYER_SIZES[0]: push_error("La cantidad de inputs debe ser igual a lo que pide la red. Red: ", LAYER_SIZES[0], "Inputs: ", len(input_vec))
	activations.clear()
	z_values.clear()
	
	activations.append(input_vec.duplicate())  # capa 0 = inputs
	
	var current : Array = input_vec.duplicate()
	
	for layer_idx in range(weights.size()):
		var w   : Array = weights[layer_idx]
		var b   : Array = biases[layer_idx]
		var out : Array = []
		var z   : Array = []
		var is_output_layer : bool = (layer_idx == weights.size() - 1)
		
		for j in range(w.size()):
			var sum : float = b[j]
			for k in range(current.size()):
				sum += w[j][k] * current[k]
			z.append(sum)
			
			if is_output_layer:
				# outputs 0-3: tanh  |  output 4: sigmoid
				if j < 4:
					out.append(_tanh(sum))
				else:
					out.append(_sigmoid(sum))
			else:
				out.append(_relu(sum))
		
		z_values.append(z)
		activations.append(out)
		current = out
	
	return current

# ─────────────────────────────────────────
#  Backpropagation para REINFORCE
#  gradients_output: Array[float] de tamaño 5
#                    = d_loss/d_output calculado por el trainer
#  learning_rate:    float
# ─────────────────────────────────────────
func backward(gradients_output: Array, learning_rate: float) -> void:
	if activations.is_empty():
		push_error("NeuralNetwork: forward() debe llamarse antes de backward().")
		return
	
	# delta[i] = gradiente respecto a z de la capa i
	var deltas : Array = []
	deltas.resize(weights.size())
	
	# --- Capa de salida ---
	var output_delta : Array = []
	var last_z       : Array = z_values[z_values.size() - 1]
	for j in range(LAYER_SIZES[LAYER_SIZES.size() - 1]):
		var d_activation : float
		if j < 4:
			d_activation = _tanh_derivative(last_z[j])
		else:
			d_activation = _sigmoid_derivative(last_z[j])
		output_delta.append(gradients_output[j] * d_activation)
	deltas[weights.size() - 1] = output_delta
	
	# --- Capas ocultas (backprop hacia atras) ---
	for layer_idx in range(weights.size() - 2, -1, -1):
		var next_delta : Array = deltas[layer_idx + 1]
		var w_next     : Array = weights[layer_idx + 1]
		var z_cur      : Array = z_values[layer_idx]
		var cur_delta  : Array = []
		
		for k in range(LAYER_SIZES[layer_idx + 1]):
			var error : float = 0.0
			for j in range(next_delta.size()):
				error += w_next[j][k] * next_delta[j]
			cur_delta.append(error * _relu_derivative(z_cur[k]))
		deltas[layer_idx] = cur_delta
	
	# --- Actualización de pesos y biases ---
	for layer_idx in range(weights.size()):
		var delta      : Array = deltas[layer_idx]
		var act_prev   : Array = activations[layer_idx]  # activacion de la capa anterior
		
		for j in range(delta.size()):
			biases[layer_idx][j] -= learning_rate * delta[j]
			for k in range(act_prev.size()):
				weights[layer_idx][j][k] -= learning_rate * delta[j] * act_prev[k]

# ─────────────────────────────────────────
#  Funciones de activacion y sus derivadas
# ─────────────────────────────────────────
func _relu(x: float) -> float:
	return max(0.0, x)

func _relu_derivative(x: float) -> float:
	return 1.0 if x > 0.0 else 0.0

func _tanh(x: float) -> float:
	# GDScript no tiene tanh nativo, lo implementamos con exp
	var e_pos : float = exp(x)
	var e_neg : float = exp(-x)
	return (e_pos - e_neg) / (e_pos + e_neg)

func _tanh_derivative(x: float) -> float:
	var t : float = _tanh(x)
	return 1.0 - t * t

func _sigmoid(x: float) -> float:
	return 1.0 / (1.0 + exp(-x))

func _sigmoid_derivative(x: float) -> float:
	var s : float = _sigmoid(x)
	return s * (1.0 - s)

# ─────────────────────────────────────────
#  Serialización de pesos (para persistencia)
# ─────────────────────────────────────────
func get_weights_data() -> Dictionary:
	return {
		"weights": weights.duplicate(true),
		"biases" : biases.duplicate(true)
	}

func set_weights_data(data: Dictionary) -> void:
	weights = data["weights"].duplicate(true)
	biases  = data["biases"].duplicate(true)
