"""Title keyword rule for the level: the rule baseline for junior or not."""
import re

LEVEL_PATTERN = re.compile(r"\bjunior\b|\bintern\b|trainee|стажер|entry[- ]level", re.IGNORECASE)


def weak_level(title: str) -> str:
    """Rule baseline for the level: junior when the title says so."""
    return "junior" if LEVEL_PATTERN.search(title) else "not_junior"