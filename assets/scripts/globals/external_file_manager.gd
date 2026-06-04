extends Node

func save_data(data:Dictionary, path:String) -> void:
	if not FileAccess.file_exists(path): print("No existe el archivo, creando uno...")
	var file = FileAccess.open(path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(data, "\t")) # Guarda la data en el json
		file.close()
		print("Datos en '%s' guardados o actualizados correctamente" % [path])
	else:
		print("Error inesperado al crear o guardar un archivo")

func read_json(path:String) -> Dictionary:
	if not FileAccess.file_exists(path): # Si la ruta no existe
		print("El archivo '%s' no existe: " % [path])
		return {} 
	
	var file = FileAccess.open(path, FileAccess.READ)
	if not file:
		print("No se pudo abrir el archivo: ", path)
		return {}
	
	var data = JSON.parse_string(file.get_as_text())
	file.close()
	
		# Verificar que el parseo fue exitoso y que es un diccionario
	if data == null:
		print("Error al parsear JSON (posible sintaxis inválida)")
		return {}
	
	if typeof(data) != TYPE_DICTIONARY:
		print("El JSON raíz no es un diccionario")
		return {} 
	
	return data
	
