extends Node

var current_reward: float = 0.0
var current_episode: int = 0
var current_step: int = 0

var players: Array[PlayerController] = []
var bullets: Array[Bullet] = []
var boss: BossController = null

var nn_outputs: Dictionary={}
