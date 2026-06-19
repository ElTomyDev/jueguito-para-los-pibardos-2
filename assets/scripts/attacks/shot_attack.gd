extends Node2D
class_name ShotAttack

@export_category("Bullet Settings")
@export var bullet_life_time: float = 2.3
@export var bullet_color: Color

@export_category("Gun Settings")
@export var gun_damage: float = 0.0
@export var rotation_speed: float = 50.0
@export var bullet_dispersion: float = 18.0
@export var fire_rate: float = 0.6
@export var is_boss: bool = false

@export_category("External Nodes")
@export var bullet: PackedScene = preload("res://assets/scenes/attacks/gun/bullet.tscn")
@export var shot_point: Node2D
@export var gun_sprite: Sprite2D = null

var character: CharacterBody2D = null

var last_shot_step: int = 0

# FIX: fire_timer arranca en fire_rate para evitar disparo inmediato en frame 1
var fire_timer: float = 0.0
var total_bullet_damage: float

func setup(body: CharacterBody2D) -> void:
	character = body
	_init_boss_values()

@warning_ignore("unused_parameter")
func update(delta: float) -> void:
	boss_shot(delta)
	players_shot(delta)

func boss_shot(delta: float) -> void:
	if (character is PlayerController) or not is_instance_valid(character):
		return
	fire_timer -= minf(delta, 0.1) 
	if fire_timer <= 0:
		if character.can_shot():  # solo dispara si la acción es "ataque"
			_shot()
			last_shot_step = GlobalVars.current_step
			character.last_shot_step = GlobalVars.current_step
		fire_timer = fire_rate  # resetea siempre, haya disparado o no

func players_shot(delta: float) -> void:
	if (character is BossController) or not is_instance_valid(character):
		return

	if is_instance_valid(gun_sprite):
		Utils.view_to(self.global_position, get_global_mouse_position(), rotation_speed, self)

	if Input.is_action_pressed("shot"):
		fire_timer -= minf(delta, 0.1) 

	if fire_timer <= 0:
		_shot()
		fire_timer = fire_rate

func _shot(custom_dir: Vector2 = Vector2.ZERO) -> void:
	var bullet_instance = bullet.instantiate()
	_set_bullet_values(bullet_instance, custom_dir)
	get_tree().get_root().add_child.call_deferred(bullet_instance)

func _set_bullet_values(bullet_instence: Bullet, custom_bullet_dir: Vector2) -> void:
	if !(bullet_instence is Bullet): push_error("'bullet_instance' debe ser una bala.")

	bullet_instence.global_position = shot_point.global_position if is_instance_valid(shot_point) else character.global_position
	bullet_instence.life_time       = bullet_life_time
	bullet_instence.damage          = _get_total_bullet_damage()
	bullet_instence.from_group      = character.bullet_from_group
	bullet_instence.group_target    = character.bullet_to_group
	bullet_instence.bullet_color    = bullet_color

	var disp_x = randf_range(-bullet_dispersion, bullet_dispersion)
	var disp_y = randf_range(-bullet_dispersion, bullet_dispersion)
	if character is PlayerController:
		bullet_instence.boss_dir   = Vector2.ZERO
		bullet_instence.player_dir = Utils.view_to(
			self.global_position,
			get_global_mouse_position() + Vector2(disp_x, disp_y),
			100.0, self, false
		) if custom_bullet_dir == Vector2.ZERO else custom_bullet_dir
	if character is BossController:
		bullet_instence.player_dir = Vector2.ZERO
		bullet_instence.boss_dir   = Vector2(cos(character.shot_angle), sin(character.shot_angle)) + Vector2(disp_x, disp_y)

func _init_boss_values() -> void:
	if !(character is BossController) or not is_instance_valid(character):
		return
	gun_damage        = 0.0
	rotation_speed    = 25.0
	fire_rate         = 0.45
	fire_timer        = fire_rate
	bullet_life_time  = 2.3
	bullet_dispersion = 0.08

func _get_total_bullet_damage() -> float:
	return gun_damage if not character else gun_damage + character.damage
