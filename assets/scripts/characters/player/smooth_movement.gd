extends Node2D
class_name PlayerSmoothMovement

var player : PlayerController

func setup(body: PlayerController) -> void:
	player = body

func update(delta: float) -> void:
	horizontal_movement(delta)

func horizontal_movement(delta: float) -> void:
	if player.dir_hor != 0: # Si la direcion horizontal (controles) es diferente de cero, se mueve.
		# Le aplica una aceleracion hacia (dir_hor) en el eje x 
		player.velocity.x = lerpf(player.velocity.x, player.max_speed * player.dir_hor, player.acceleration_speed * delta)
	else: # Sino, no se mueve.
		# Desacelera el jugador hasta que su velocidad sea igual a 0
		player.velocity.x = lerpf(player.velocity.x, 0.0, player.deceleration_speed * delta)
