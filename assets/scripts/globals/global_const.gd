extends Node

const MAX_STEP_FOR_EPISODE: int = 600

const SAVE_MODEL_PATH: String = "res://assets/train_data/boss_brain.json"
const SAVE_BEST_MODEL_PATH = "res://assets/train_data/boss_brain_best.json"

const BEST_TRAIN_DATA_PATH= "res://assets/train_data/train_data.json"

const REWARD_CSV_PATH= "res://assets/train_data/reward_data.csv"

const REWARD_WINDOW: int = 10 # Maxima acumulacion de episodios antes de obtener promedio
