extends Node2D
class_name Bullet

var bullet_color: Color = Color.RED

var speed: float
var life_time: float
var dispersion: float
var dir_to_mirror: Vector2
var damage: float

var custom_dir: Vector2 = Vector2.ZERO

var from_group: StringName
var group_target: StringName
var target:Area2D = null

func _ready() -> void:
	var disp_x = randf_range(-dispersion, dispersion)
	var disp_y = randf_range(-dispersion, dispersion)
	if custom_dir != Vector2.ZERO:
		# Bala del boss: usa la dirección que decidió la red
		dir_to_mirror = custom_dir + Vector2(disp_x, disp_y)
	else:
		# Bala del jugador: apunta al mouse
		dir_to_mirror = Utils.view_to(
			self.global_position,
			get_global_mouse_position() + Vector2(disp_x, disp_y),
			100.0, self, false
		)
	GlobalVars.bullets.append(self) # Agrega la bala a las variables globales para usarlas en el simulation_manager.gd

func _process(delta: float) -> void:
	_dead_if_can(delta)
	queue_redraw()

func _physics_process(delta: float) -> void:
	move_bullet(delta)

func move_bullet(delta:float) -> void:
	self.position += dir_to_mirror.normalized() * speed * delta 

func _dead_if_can(delta: float) -> void:
	if life_time <= 0:
		queue_free()
		GlobalVars.bullet.pop_at(GlobalVars.bullet.find(self))
	life_time -= delta

func _draw() -> void:
	draw_line(Vector2.ZERO, Vector2(8,0), bullet_color, 3.5)

func _on_area_entered(area: Area2D) -> void:
	if area.is_in_group(group_target):
		target = area
		target.apply_damage(damage)
		GlobalVars.bullet.pop_at(GlobalVars.bullet.find(self)) # Elimina la bala de la variable global
		queue_free()

func _on_area_exited(area: Area2D) -> void:
	if area.is_in_group(group_target):
		target = null
