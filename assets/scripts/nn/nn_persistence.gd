extends Node
class_name NNPersistence

# ─────────────────────────────────────────
#  Guarda y carga los pesos de la red en un
#  archivo JSON en user:// (persistente entre runs)
# ─────────────────────────────────────────

const SAVE_PATH : String = "res://assets/train_data/boss_brain.json"

# ─────────────────────────────────────────
#  Guarda los pesos de la red en disco
# ─────────────────────────────────────────
func save(nn: NeuralNetwork, trainer: NNTrainer) -> void:
	# Construimos el diccionario con la estructura de la red
	var data : Dictionary = {
		"weights": nn.weights,
		"biases": nn.biases,
		"sigma": trainer.current_sigma
	}
	
	var json_str   : String     = JSON.stringify(data)
	var file       : FileAccess = FileAccess.open(SAVE_PATH, FileAccess.WRITE)
	
	if file == null:
		push_error("NNPersistence: no se pudo abrir el archivo para escritura: " + SAVE_PATH)
		return
	
	file.store_string(json_str)
	file.close()
	print("NNPersistence: pesos y sigma guardados en ", SAVE_PATH)

# ─────────────────────────────────────────
#  Carga los pesos desde disco y los aplica a la red
#  Retorna true si cargó con éxito, false si no había archivo
# ─────────────────────────────────────────
func load_into(nn: NeuralNetwork, trainer: NNTrainer) -> bool:
	if not FileAccess.file_exists(SAVE_PATH):
		print("NNPersistence: no existe archivo previo, empezando con pesos aleatorios.")
		return false
	
	var file : FileAccess = FileAccess.open(SAVE_PATH, FileAccess.READ)
	if file == null:
		push_error("NNPersistence: no se pudo abrir el archivo para lectura: " + SAVE_PATH)
		return false
		
	var json_str : String = file.get_as_text()
	file.close()
	
	var json : JSON = JSON.new()
	var parse_error : int = json.parse(json_str)
	
	if parse_error != OK:
		push_error("NNPersistence: error al parsear JSON: " + json.get_error_message())
		return false
	
	var data : Dictionary = json.get_data()
	
	if not _validate_data(data, nn):
		push_warning("NNPersistence: estructura de pesos incompatible o corrupta. Se inicia con pesos aleatorios.")
		return false
	
	# Asignamos los datos validados a la red y al trainer
	nn.weights = data["weights"]
	nn.biases = data["biases"]
	
	if data.has("sigma"):
		trainer.current_sigma = data["sigma"]
		
	print("NNPersistence: pesos y sigma cargados exitosamente desde ", SAVE_PATH)
	return true

# ─────────────────────────────────────────
#  Valida que el archivo guardado sea compatible
#  con la arquitectura actual de la red
# ─────────────────────────────────────────
func _validate_data(data: Dictionary, nn: NeuralNetwork) -> bool:
	if not data.has("weights") or not data.has("biases"):
		return false
	
	var saved_weights : Array = data["weights"]
	var saved_biases  : Array = data["biases"]
	
	# Verificar que la cantidad de capas coincida
	if saved_weights.size() != nn.weights.size() or saved_biases.size() != nn.biases.size():
		return false
	
	# Validar las dimensiones de cada capa de pesos y biases
	for i in range(saved_weights.size()):
		if saved_weights[i].size() != nn.weights[i].size():
			return false
		if saved_biases[i].size() != nn.biases[i].size():
			return false
		# Validar neuronas de entrada de la capa (fan_in)
		if saved_weights[i].size() > 0:
			if saved_weights[i][0].size() != nn.weights[i][0].size():
				return false
	
	return true

# ─────────────────────────────────────────
#  Borra los pesos guardados (útil para resetear entrenamiento)
# ─────────────────────────────────────────
func clear_saved_data() -> void:
	if FileAccess.file_exists(SAVE_PATH):
		var dir = DirAccess.open("res://")
		var err = dir.remove(SAVE_PATH)
		if err == OK:
			print("NNPersistence: Archivo de guardado eliminado con éxito.")
		else:
			push_error("NNPersistence: No se pudo eliminar el archivo de guardado. Código de error: ", err)
