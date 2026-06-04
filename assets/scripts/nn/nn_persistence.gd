extends RefCounted
class_name NNPersistence

func save_network(nn: NeuralNetwork, path: String) -> void:
	var data: Dictionary = {
		"W1": nn.W1,
		"b1": nn.b1,
		"W_actor": nn.W_actor,
		"b_actor": nn.b_actor,
		"W_critic": nn.W_critic,
		"b_critic": nn.b_critic
	}
	ExternalFileManager.save_data(data, path)

func load_network(nn: NeuralNetwork, path: String) -> bool:
	
	var data = ExternalFileManager.read_json(path)
	if typeof(data) != TYPE_DICTIONARY or not data.has("W1") or not data.has("b1"):
		print("[NNPersistence] Estructura inválida. Usando pesos aleatorios.")
		return false
	
	# Validación de nan antes de cargar
	if _matrix_has_nan(data["W1"]) or _array_has_nan(data["b1"]):
		print("[NNPersistence] Cerebro corrupto (NaN detectado). Usando pesos aleatorios.")
		return false
	
	nn.W1      = data["W1"]
	nn.b1      = data["b1"]
	nn.W_actor = data["W_actor"]
	nn.b_actor = data["b_actor"]
	nn.W_critic = data["W_critic"]
	nn.b_critic = data["b_critic"]
	print("[NNPersistence] Cerebro cargado correctamente.")
	return true
	
func _matrix_has_nan(matrix: Array) -> bool:
	for row in matrix:
		if _array_has_nan(row):
			return true
	return false

func _array_has_nan(arr: Array) -> bool:
	for val in arr:
		if typeof(val) == TYPE_FLOAT and is_nan(val):
			return true
	return false
