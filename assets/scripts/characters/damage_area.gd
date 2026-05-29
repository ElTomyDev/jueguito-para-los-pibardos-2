extends Area2D
class_name DamageArea

var char:CharacterBody2D

func setup(body: CharacterBody2D) -> void:
	char = body

func apply_damage(damage:float) -> void:
	char.health -= damage
