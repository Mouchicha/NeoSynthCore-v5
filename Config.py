CONFIG = {
"d_model": 256,
"vocab_size": 10000,
"emo_classes": 6,
"stage_classes": 5,
"cultures": 3,
"seq_len": 50,
"batch_size": 128,
"epochs": 5,
"lr": 3e-4,
"device": "cuda",
"lambdas": {
"lm": 1.0, "emo": 0.5, "stage": 0.5,
"safe": 1.0, "culture": 0.3, "vib": 0.1, "heal": 0.2
}
}