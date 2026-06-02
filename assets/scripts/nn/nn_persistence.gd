extends RefCounted
class_name NNPersistence

const SAVE_PATH: String = "res://assets/train_data/boss_brain.json"

func save_network(nn: NeuralNetwork) -> void:
	var data: Dictionary = {
		"W1": nn.W1,
		"b1": nn.b1,
		"W_actor": nn.W_actor,
		"b_actor": nn.b_actor,
		"W_critic": nn.W_critic,
		"b_critic": nn.b_critic
	}
	
	var file = FileAccess.open(SAVE_PATH, FileAccess.WRITE)
	if file:
		var json_string: String = JSON.stringify(data)
		file.store_string(json_string)
		file.close()
		print("[NNPersistence] Cerebro del jefe guardado con éxito.")
	else:
		push_error("[NNPersistence] No se pudo abrir el archivo para guardar.")

func load_network(nn: NeuralNetwork) -> bool:
	if not FileAccess.file_exists(SAVE_PATH):
		print("[NNPersistence] No se encontró un cerebro previo. Usando pesos aleatorios.")
		return false
		
	var file = FileAccess.open(SAVE_PATH, FileAccess.READ)
	if file:
		var json_string: String = file.get_as_text()
		file.close()
		
		var json = JSON.new()
		var error = json.parse(json_string)
		if error == OK:
			var data = json.get_data()
			# Verificación estricta de que el diccionario sea válido y contenga las claves de la red
			if typeof(data) == TYPE_DICTIONARY and data.has("W1") and data.has("b1"):
				nn.W1 = data["W1"]
				nn.b1 = data["b1"]
				nn.W_actor = data["W_actor"]
				nn.b_actor = data["b_actor"]
				nn.W_critic = data["W_critic"]
				nn.b_critic = data["b_critic"]
				print("[NNPersistence] Cerebro del jefe cargado de forma persistente.")
				return true
			else:
				print("[NNPersistence] El JSON cargado no tiene la estructura correcta. Usando pesos aleatorios.")
				return false
			
	push_error("[NNPersistence] Error crítico al parsear el archivo de guardado.")
	return false
