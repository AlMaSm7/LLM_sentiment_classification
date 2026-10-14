from llm_project.utils.constants import INSTRUCTION


def build_prompt(
    text: str,
    instruction: str = INSTRUCTION,
) -> str:
    """Build the common emotion-classification prompt."""

    return f"{instruction}\n\nText:\n{text}\n"
