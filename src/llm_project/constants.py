XED_BASE_URL = "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections"

EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]

EMOTION_MAP = {str(index + 1): emotion for index, emotion in enumerate(EMOTIONS)}

INSTRUCTION = (
    "Identify all emotions expressed in the text. "
    "Return only a JSON list using at least one or more of these labels: "
    "anger, anticipation, disgust, fear, joy, sadness, surprise, trust."
)
