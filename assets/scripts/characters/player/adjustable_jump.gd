extends Node2D
class_name PlayerAdjustableJump

var player : PlayerController

var is_jumping: bool = false # Indica si el jugador esta en proceso de salto
var jump_time: float = 0.0   # Temporizador para saber por cuanto tiempo se preciona la tecla de salto.

func setup(body: PlayerController) -> void:
	player = body

func update(delta: float) -> void:
	apply_gravity(delta)
	apply_adjustable_jump(delta)

func apply_adjustable_jump(delta: float) -> void:
	if can_start_jump(): # Si esta en el suelo y preciono tecla de salto
		is_jumping = true 
		jump_time = 0.0
	
	if player.controls.input_jump_pressed() and is_jumping: # Si se mantiene apretado el boton y esta en prceso de salto
		if jump_time < player.max_jump_time: # Si el tiempo saltando no supera el tiempo maximo de salto.
			jump_time += delta # Se acumula el tiempo que esta saltando
			jump(delta)
	else:
		is_jumping = false

func can_start_jump() -> bool: # Indica si el jugador puede o no inicial el salto
	return player.controls.input_jump_just_pressed() and player.is_on_floor()

func jump(delta) -> void:
	player.velocity.y -= player.jump_force * delta

func apply_gravity(delta: float) -> void:
	"""
	Le aplica la gravedad al jugador en el eje Y hacia abajo limitando la misma 
	para que su velocidad no incremente hacia el 'infinito'.
	"""
	var max_fall_speed = 700 # La velocidad maxima de caida.
	if not player.is_on_floor():
		player.velocity.y = min(player.velocity.y + player.gravity * delta, max_fall_speed) # Aplica la gravedad para el eje Y
