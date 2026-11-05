from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_reviews.csv"


@pytest.fixture(scope="session")
def fixture_csv() -> Path:
    return FIXTURE
