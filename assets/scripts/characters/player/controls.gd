extends Node2D
class_name  PlayerControls

var player : PlayerController

@export_category("Movement Controls")
@export var right_control : StringName  # Tecla movimiento hacia la derecha
@export var left_control : StringName   # Tecla movimiento hacia la izquierda

@export_category("Jump Control")
@export var jump_control : StringName   # Tecla para el salto

func setup(body: PlayerController) -> void:
	player = body

func update() -> void:
	update_horizontal_directions()

func update_horizontal_directions() -> void:
	# Actualiza la direccion horizontal del jugador: 
		#  1 para la derecha
		#  0 para estar quieto
		# -1 para la izquierda
	@warning_ignore("narrowing_conversion")
	player.dir_hor = Input.get_action_strength(right_control) - Input.get_action_strength(left_control)

func input_jump_just_pressed() -> bool:
	"""
	Indica si la tecla de salto (jump_control) se preciono, es verdadero solo al
	instante en el que se preciona la tecla.
	"""
	return Input.is_action_just_pressed(jump_control)

func input_jump_pressed() -> bool:
	"""
	Indica si la tecla de salto (jump_control) esta siendo precionada, es verdadero
	siempre que la tecla se este precionando.
	"""
	return Input.is_action_pressed(jump_control)
