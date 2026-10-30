import pandas as pd
import pytest

from data.schema import detect_language, level_from_years
from rules.levels import weak_level
from rules.roles import weak_role

@pytest.mark.parametrize("title, role", [
    ("Senior Data Analyst", "Data Analyst"),
    ("Data Scientist", "Data Scientist"),
    ("ML Engineer", "ML / AI Engineer"),
    ("Python Backend Engineer", "Python Backend"),
    ("Java Backend Developer", "Other"),
    ("Business Analyst (1C)", "Other"),
    ("Senior QA Automation Engineer (Python)", "Other"),
])
def test_weak_role(title, role):
    assert weak_role(title) == role

@pytest.mark.parametrize("title, level", [
    ("Junior Python Developer", "junior"),
    ("Trainee Data Analyst", "junior"),
    ("Senior Data Scientist", "not_junior"),
    ("International Sales Manager", "not_junior"),
])
def test_weak_level(title, level):
    assert weak_level(title) == level


@pytest.mark.parametrize("years, level", [(0.5, "junior"), (1, "junior"), (2, "not_junior")])
def test_level_from_years(years, level):
    assert level_from_years(years) == level

def test_level_from_years_unknown():
    assert level_from_years(None) is None
    assert level_from_years(float("nan")) is None

def test_detect_language():
    assert detect_language("Шукаємо аналітика даних") == "uk"
    assert detect_language("We are looking for a data analyst") == "en"