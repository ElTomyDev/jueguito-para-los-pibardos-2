extends Node2D
class_name BossCollision

var boss: BossController

func setup(body: BossController) -> void:
	boss = body

func _apply_damage(damage:float) -> void:
	boss.health -= damage

# --------------------------------------
# --- Colision con balas del jugador ---
# --------------------------------------
func _on_damage_area_body_entered(bullet: Node2D) -> void:
	if is_instance_valid(bullet):
		if bullet.is_in_group("Bullets") and bullet.group_target == "Boss":
			_apply_damage(bullet.damage)
			bullet.delete_bullet(boss, boss.near_player)

func _on_bullet_detector_body_entered(bullet: Node2D) -> void:
	if not (is_instance_valid(bullet) and bullet.is_in_group("Bullets") and bullet.group_target == "Boss"):
		return
	# Busca el primer slot libre (null) para insertar
	for i in range(boss.bullets_detected.size()):
		if boss.bullets_detected[i] == null:
			boss.bullets_detected[i] = bullet
			return
	# Si no hay slot libre, reemplaza el primero (FIFO)
	boss.bullets_detected[0] = bullet

func _on_bullet_detector_body_exited(bullet: Node2D) -> void:
	if not is_instance_valid(bullet):
		return
	var idx = boss.bullets_detected.find(bullet)
	if idx != -1:
		boss.bullets_detected[idx] = null
