EMOTIONS = [
    "Like",
    "Happiness",
    "Anger",
    "Disgust",
    "Fear",
    "Sadness",
    "Surprise",
]

LABEL_TO_ID = {label: i for i, label in enumerate(EMOTIONS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
