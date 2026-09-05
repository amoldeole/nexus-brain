import pytest

from nexus_brain.brain import Brain
from nexus_brain.core.bus import EventBus


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def brain() -> Brain:
    b = Brain()
    b.config.learning.reflect_every_n_cycles = 10_000  # tests trigger reflection explicitly
    return b
