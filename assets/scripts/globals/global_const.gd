extends Node

const MAX_STEP_FOR_EPISODE: int = 800

const SAVE_PATH_MODEL: String = "res://assets/train_data/boss_brain.json"
const SAVE_PATH_BEST_MODEL = "res://assets/train_data/boss_brain_best.json"

const BEST_TRAIN_DATA_PATH= "res://assets/train_data/last_train_data.json"

const REWARD_WINDOW: int = 10 # Maxima acumulacion de episodios antes de obtener promedio
