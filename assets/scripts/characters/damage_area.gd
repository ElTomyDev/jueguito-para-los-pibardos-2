extends Area2D
class_name DamageArea

var character:CharacterBody2D

func setup(body: CharacterBody2D) -> void:
	character = body

func apply_damage(damage:float) -> void:
	character.health -= damage
