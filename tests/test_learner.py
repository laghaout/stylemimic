from learners import learner as lea
from learners import utilities as util
import pytest

learner = lea.Learner()


def test_learner_has_report() -> None:
    """Test that the learner object has a report attribue."""

    learner()
    assert hasattr(learner, "report")


def test_learner_has_data() -> None:
    """Test that the learner object has a data attribue."""

    learner()
    assert hasattr(learner, "data")


def test_utilities__version_table() -> None:
    """Test that we can obtain the version table."""

    assert isinstance(util.version_table(), dict)


def test_something_else() -> None:
    """Test that we raise an error under some condition."""

    with pytest.raises(Exception):
        if True:
            raise Exception("Running some error code.")
