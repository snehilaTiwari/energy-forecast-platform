# tests/test_promote.py
import pytest
from unittest.mock import MagicMock
from src.training.promote import (
    get_production_rmse,
    get_best_challenger_rmse_and_version,
    should_promote,
    IMPROVEMENT_THRESHOLD,
)

def _mock_version(version: str, run_id: str, stage: str = "None") -> MagicMock:
    v = MagicMock()
    v.version = version
    v.run_id = run_id
    v.current_stage = stage
    return v

def _mock_run(rmse: float) -> MagicMock:
    r = MagicMock()
    r.data.metrics = {"rmse": rmse}
    return r

def test_get_production_rmse_returns_none_when_no_production_model():
    client = MagicMock()
    client.search_model_versions.return_value = []
    assert get_production_rmse(client, "EnergyForecast") is None

def test_get_production_rmse_returns_rmse_from_run():
    client = MagicMock()
    client.search_model_versions.return_value = [
        _mock_version("2", "run-abc", stage="Production")
    ]
    client.get_run.return_value = _mock_run(0.25)
    result = get_production_rmse(client, "EnergyForecast")
    assert result == pytest.approx(0.25)

def test_get_best_challenger_rmse_and_version_picks_lowest_rmse():
    client = MagicMock()
    # Two challengers in "None" stage plus one in Production (should be ignored)
    client.search_model_versions.return_value = [
        _mock_version("3", "run-1", stage="None"),
        _mock_version("4", "run-2", stage="None"),
        _mock_version("2", "run-0", stage="Production"),
    ]
    client.get_run.side_effect = [_mock_run(0.30), _mock_run(0.22)]
    version, rmse = get_best_challenger_rmse_and_version(client, "EnergyForecast")
    assert version == "4"
    assert rmse == pytest.approx(0.22)

def test_get_best_challenger_ignores_production_versions():
    client = MagicMock()
    client.search_model_versions.return_value = [
        _mock_version("1", "run-1", stage="Production"),
    ]
    result = get_best_challenger_rmse_and_version(client, "EnergyForecast")
    assert result is None

def test_should_promote_when_improvement_exceeds_threshold():
    # 10% improvement > 2% threshold
    assert should_promote(prod_rmse=0.30, challenger_rmse=0.27) is True

def test_should_promote_rejects_insufficient_improvement():
    # 1% improvement < 2% threshold
    assert should_promote(prod_rmse=0.30, challenger_rmse=0.297) is False

def test_should_promote_when_no_production_model():
    assert should_promote(prod_rmse=None, challenger_rmse=0.25) is True
