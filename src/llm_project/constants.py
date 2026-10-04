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
    "Return only a JSON list using one or more of these labels: "
    "anger, anticipation, disgust, fear, joy, sadness, surprise, trust."
)

INSTRUCTION_QWEN = "Identify all emotions expressed in the text. Return ONLY a valid JSON list of strings using one or more of these labels: anger, anticipation, disgust, fear, joy, sadness, surprise, trust. Output exactly the bracketed list and nothing else. Do not include any explanations, introductory text, or markdown code blocks."
