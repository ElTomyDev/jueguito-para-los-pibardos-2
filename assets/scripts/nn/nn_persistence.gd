extends RefCounted
class_name NNPersistence

const SAVE_PATH: String = "res://assets/train_data/boss_brain.json"
const SAVE_PATH_BEST = "res://assets/train_data/boss_brain_best.json"

func save_network(nn: NeuralNetwork, path: String) -> void:
	var data: Dictionary = {
		"W1": nn.W1,
		"b1": nn.b1,
		"W_actor": nn.W_actor,
		"b_actor": nn.b_actor,
		"W_critic": nn.W_critic,
		"b_critic": nn.b_critic
	}
	
	var file = FileAccess.open(path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(data))
		file.close()
		print("[NNPersistence] Guardado en ", path)

func load_network(nn: NeuralNetwork) -> bool:
	if not FileAccess.file_exists(SAVE_PATH):
		print("[NNPersistence] No se encontró cerebro previo. Usando pesos aleatorios.")
		return false
		
	var file = FileAccess.open(SAVE_PATH, FileAccess.READ)
	if not file:
		push_error("[NNPersistence] No se pudo abrir el archivo.")
		return false

	var json_string: String = file.get_as_text()
	file.close()
	
	var json = JSON.new()
	if json.parse(json_string) != OK:
		push_error("[NNPersistence] Error al parsear JSON.")
		return false
		
	var data = json.get_data()
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
