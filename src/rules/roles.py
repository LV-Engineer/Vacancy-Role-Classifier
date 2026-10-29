"""Title keyword rule for the role: weak labels for LinkedIn and the rule baseline."""
import re

ROLE_PATTERNS = [
    ("Other", r"business analyst|\bqa\b|test automation|automation|quality assurance"),
    ("ML / AI Engineer", r"machine learning|\bml\b|ai engineer|mlops|deep learning|\bllm\b"),
    ("Data Scientist", r"data scientist"),
    ("Data Analyst", r"data analyst|bi analyst|bi developer"),
    ("Python Backend", r"python|django|fastapi|flask"),
]


def weak_role(title: str) -> str:
    """Weak label from title keywords. Used for train and as a pre-label for manual work."""
    lowered = title.lower()
    for role, pattern in ROLE_PATTERNS:
        if re.search(pattern, lowered):
            return role
    return "Other"
