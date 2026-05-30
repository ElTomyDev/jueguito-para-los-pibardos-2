extends Node
class_name NNPersistence

# ─────────────────────────────────────────
#  Guarda y carga los pesos de la red en un
#  archivo JSON en user:// (persistente entre runs)
# ─────────────────────────────────────────

const SAVE_PATH : String = "res://assets/train_data/"

# ─────────────────────────────────────────
#  Guarda los pesos de la red en disco
# ─────────────────────────────────────────
func save(nn: NeuralNetwork) -> void:
	var data       : Dictionary = nn.get_weights_data()
	var json_str   : String     = JSON.stringify(data)
	var file       : FileAccess = FileAccess.open(SAVE_PATH, FileAccess.WRITE)
	
	if file == null:
		push_error("NNPersistence: no se pudo abrir el archivo para escritura: " + SAVE_PATH)
		return
	
	file.store_string(json_str)
	file.close()
	print("NNPersistence: pesos guardados en ", SAVE_PATH)

# ─────────────────────────────────────────
#  Carga los pesos desde disco y los aplica a la red
#  Retorna true si cargó con éxito, false si no había archivo
# ─────────────────────────────────────────
func load_into(nn: NeuralNetwork) -> bool:
	if not FileAccess.file_exists(SAVE_PATH):
		print("NNPersistence: no existe archivo previo, empezando con pesos aleatorios.")
		return false
	
	var file : FileAccess = FileAccess.open(SAVE_PATH, FileAccess.READ)
	if file == null:
		push_error("NNPersistence: no se pudo abrir el archivo para lectura: " + SAVE_PATH)
		return false
	
	var json_str : String     = file.get_as_text()
	file.close()
	
	var json        : JSON       = JSON.new()
	var parse_error : int        = json.parse(json_str)
	
	if parse_error != OK:
		push_error("NNPersistence: error al parsear JSON: " + json.get_error_message())
		return false
	
	var data : Dictionary = json.get_data()
	
	if not _validate_data(data, nn):
		push_warning("NNPersistence: estructura de pesos incompatible, reiniciando con pesos aleatorios.")
		return false
	
	nn.set_weights_data(data)
	print("NNPersistence: pesos cargados desde ", SAVE_PATH)
	return true

# ─────────────────────────────────────────
#  Valida que el archivo guardado sea compatible
#  con la arquitectura actual de la red
# ─────────────────────────────────────────
func _validate_data(data: Dictionary, nn: NeuralNetwork) -> bool:
	if not data.has("weights") or not data.has("biases"):
		return false
	
	var saved_weights : Array = data["weights"]
	if saved_weights.size() != nn.weights.size():
		return false
	
	for i in range(saved_weights.size()):
		if saved_weights[i].size() != nn.weights[i].size():
			return false
		if saved_weights[i].size() > 0:
			if saved_weights[i][0].size() != nn.weights[i][0].size():
				return false
	
	return true

# ─────────────────────────────────────────
#  Borra los pesos guardados (útil para resetear el aprendizaje)
# ─────────────────────────────────────────
func reset_saved_data() -> void:
	if FileAccess.file_exists(SAVE_PATH):
		DirAccess.remove_absolute(SAVE_PATH)
		print("NNPersistence: datos de aprendizaje reseteados.")
