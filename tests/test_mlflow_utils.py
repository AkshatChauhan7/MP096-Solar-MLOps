"""Unit tests for MLflow utilities."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import numpy as np

from src.mlflow_utils import (
    setup_mlflow,
    get_or_create_experiment,
    log_params,
    log_metrics,
    log_model,
    register_model,
    set_model_alias,
    transition_model_stage,
    get_latest_model_version,
    get_model_by_stage,
    log_artifacts,
    log_figure,
    get_run_info,
    search_runs,
    log_dataset_info,
    create_model_signature,
    log_confusion_matrix,
    MLflowTracker,
    promote_model_to_production,
    archive_old_production_models,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)


class TestSetupMLflow:
    """Tests for setup_mlflow function."""

    @patch("src.mlflow_utils.mlflow.set_tracking_uri")
    @patch("src.mlflow_utils.mlflow.set_experiment")
    def test_setup_calls_correct_functions(self, mock_set_experiment, mock_set_uri):
        setup_mlflow("sqlite:///test.db")
        mock_set_uri.assert_called_once_with("sqlite:///test.db")
        mock_set_experiment.assert_called_once_with(EXPERIMENT_NAME)

    def test_default_tracking_uri(self):
        with patch("src.mlflow_utils.mlflow.set_tracking_uri") as mock_set_uri, \
                patch("src.mlflow_utils.mlflow.set_experiment"):
            setup_mlflow()
            mock_set_uri.assert_called_once_with(MLFLOW_TRACKING_URI)


class TestGetOrCreateExperiment:
    """Tests for get_or_create_experiment function."""

    @patch("src.mlflow_utils.MlflowClient")
    def test_returns_existing_experiment_id(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_client.get_experiment_by_name.return_value = mock_experiment

        exp_id = get_or_create_experiment("test_exp")

        assert exp_id == "123"
        mock_client.get_experiment_by_name.assert_called_once_with("test_exp")
        mock_client.create_experiment.assert_not_called()

    @patch("src.mlflow_utils.MlflowClient")
    def test_creates_new_experiment(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_experiment_by_name.return_value = None
        mock_client.create_experiment.return_value = "456"

        exp_id = get_or_create_experiment("new_exp")

        assert exp_id == "456"
        mock_client.create_experiment.assert_called_once_with("new_exp")


class TestLogParamsAndMetrics:
    """Tests for log_params and log_metrics functions."""

    @patch("src.mlflow_utils.mlflow.log_params")
    def test_log_params(self, mock_log_params):
        log_params({"param1": "value1", "param2": 2})
        mock_log_params.assert_called_once_with({"param1": "value1", "param2": 2})

    @patch("src.mlflow_utils.mlflow.log_metrics")
    def test_log_metrics(self, mock_log_metrics):
        log_metrics({"mae": 0.5, "r2": 0.9}, step=10)
        mock_log_metrics.assert_called_once_with({"mae": 0.5, "r2": 0.9}, step=10)


class TestLogModel:
    """Tests for log_model function."""

    @patch("src.mlflow_utils.mlflow.sklearn.log_model")
    def test_log_model_returns_uri(self, mock_log_model):
        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/abc123/model"
        mock_log_model.return_value = mock_model_info

        model = MagicMock()
        uri = log_model(model, artifact_path="model", registered_model_name="test_model")

        assert uri == "runs:/abc123/model"
        mock_log_model.assert_called_once()


class TestRegisterModel:
    """Tests for register_model function."""

    @patch("src.mlflow_utils.mlflow.register_model")
    def test_register_model(self, mock_register):
        mock_mv = MagicMock()
        mock_register.return_value = mock_mv

        result = register_model("runs:/abc123/model", "test_model")

        assert result == mock_mv
        mock_register.assert_called_once_with(model_uri="runs:/abc123/model", name="test_model")

    @patch("src.mlflow_utils.MlflowClient")
    def test_sets_champion_alias(self, mock_client_class):
        client = MagicMock()
        mock_client_class.return_value = client

        set_model_alias("test_model", "champion", 1)

        client.set_registered_model_alias.assert_called_once_with("test_model", "champion", "1")


class TestTransitionModelStage:
    """Tests for transition_model_stage function."""

    @patch("src.mlflow_utils.MlflowClient")
    def test_transition_stage(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        transition_model_stage("test_model", 1, "Production", archive_existing_versions=True)

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="test_model",
            version=1,
            stage="Production",
            archive_existing_versions=True,
        )


class TestGetLatestModelVersion:
    """Tests for get_latest_model_version function."""

    @patch("src.mlflow_utils.MlflowClient")
    def test_returns_max_version(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        v1 = MagicMock()
        v1.version = 1
        v2 = MagicMock()
        v2.version = 3
        v3 = MagicMock()
        v3.version = 2
        mock_client.search_model_versions.return_value = [v1, v2, v3]

        version = get_latest_model_version("test_model")

        assert version == 3

    @patch("src.mlflow_utils.MlflowClient")
    def test_returns_none_if_no_versions(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.search_model_versions.return_value = []

        version = get_latest_model_version("test_model")

        assert version is None


class TestGetModelByStage:
    """Tests for get_model_by_stage function."""

    @patch("src.mlflow_utils.mlflow.sklearn.load_model")
    @patch("src.mlflow_utils.MlflowClient")
    def test_loads_model_from_stage(self, mock_client_class, mock_load_model):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_version = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_version]

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        result = get_model_by_stage("test_model", "Production")

        assert result == mock_model
        mock_load_model.assert_called_once_with("models:/test_model/Production")

    @patch("src.mlflow_utils.MlflowClient")
    def test_returns_none_if_no_version_in_stage(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_latest_versions.return_value = []

        result = get_model_by_stage("test_model", "Production")

        assert result is None


class TestLogDatasetInfo:
    """Tests for log_dataset_info function."""

    @patch("src.mlflow_utils.mlflow.log_params")
    def test_logs_all_dataset_info(self, mock_log_params):
        log_dataset_info(
            train_path="data/train.csv",
            test_path="data/test.csv",
            train_shape=(100, 10),
            test_shape=(20, 10),
            features=["a", "b", "c"],
            target="y",
        )

        call_args = mock_log_params.call_args[0][0]
        assert call_args["train_path"] == "data/train.csv"
        assert call_args["test_path"] == "data/test.csv"
        assert call_args["train_rows"] == 100
        assert call_args["train_cols"] == 10
        assert call_args["test_rows"] == 20
        assert call_args["test_cols"] == 10
        assert call_args["n_features"] == 3
        assert call_args["target"] == "y"
        assert call_args["features"] == '["a", "b", "c"]'


class TestCreateModelSignature:
    """Tests for create_model_signature function."""

    @patch("src.mlflow_utils.infer_signature")
    def test_creates_signature(self, mock_infer):
        mock_signature = MagicMock()
        mock_infer.return_value = mock_signature

        X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y_pred = np.array([10, 20])

        result = create_model_signature(X_train, y_pred)

        assert result == mock_signature
        mock_infer.assert_called_once_with(X_train, y_pred)


class TestMLflowTracker:
    """Tests for MLflowTracker context manager."""

    @patch("src.mlflow_utils.mlflow.start_run")
    @patch("src.mlflow_utils.mlflow.end_run")
    @patch("src.mlflow_utils.mlflow.set_experiment")
    @patch("src.mlflow_utils.setup_mlflow")
    def test_context_manager(self, mock_setup, mock_set_exp, mock_end, mock_start):
        mock_run = MagicMock()
        mock_start.return_value = mock_run

        with MLflowTracker(run_name="test_run", tags={"key": "value"}) as tracker:
            assert tracker.run == mock_run

        mock_setup.assert_called_once()
        mock_set_exp.assert_called_once_with(EXPERIMENT_NAME)
        mock_start.assert_called_once_with(run_name="test_run", tags={"key": "value"})
        mock_end.assert_called_once_with(status="FINISHED")

    @patch("src.mlflow_utils.mlflow.start_run")
    @patch("src.mlflow_utils.mlflow.end_run")
    @patch("src.mlflow_utils.mlflow.set_experiment")
    @patch("src.mlflow_utils.setup_mlflow")
    def test_context_manager_marks_failed_run(self, mock_setup, mock_set_exp, mock_end, mock_start):
        mock_start.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="training failed"):
            with MLflowTracker(run_name="failed_run"):
                raise RuntimeError("training failed")

        mock_end.assert_called_once_with(status="FAILED")

    @patch("src.mlflow_utils.mlflow.start_run")
    @patch("src.mlflow_utils.mlflow.end_run")
    @patch("src.mlflow_utils.mlflow.set_experiment")
    @patch("src.mlflow_utils.setup_mlflow")
    def test_log_methods_delegate(self, mock_setup, mock_set_exp, mock_end, mock_start):
        mock_run = MagicMock()
        mock_start.return_value = mock_run

        with patch("src.mlflow_utils.log_params") as mock_log_params, \
             patch("src.mlflow_utils.log_metrics") as mock_log_metrics, \
             patch("src.mlflow_utils.log_model") as mock_log_model, \
             patch("src.mlflow_utils.mlflow.log_artifact") as mock_log_artifact:

            with MLflowTracker() as tracker:
                tracker.log_params({"p": 1})
                tracker.log_metrics({"m": 0.5})
                tracker.log_model(MagicMock())
                tracker.log_artifact("file.txt")

            mock_log_params.assert_called_once_with({"p": 1})
            # log_metrics is called with step=None by default
            mock_log_metrics.assert_called_once_with({"m": 0.5}, None)
            mock_log_model.assert_called_once()
            assert "registered_model_name" not in mock_log_model.call_args.kwargs
            mock_log_artifact.assert_called_once_with("file.txt", None)


class TestPromoteModelToProduction:
    """Tests for promote_model_to_production function."""

    @patch("src.mlflow_utils.transition_model_stage")
    @patch("src.mlflow_utils.MlflowClient")
    def test_promotes_best_staging_model(self, mock_client_class, mock_transition):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create mock versions in staging
        v1 = MagicMock()
        v1.version = 1
        v1.run_id = "run1"
        v2 = MagicMock()
        v2.version = 2
        v2.run_id = "run2"
        mock_client.get_latest_versions.return_value = [v1, v2]

        # Create mock runs with metrics
        run1 = MagicMock()
        run1.data.metrics = {"mae": 0.5}
        run2 = MagicMock()
        run2.data.metrics = {"mae": 0.3}  # better
        mock_client.get_run.side_effect = [run1, run2]

        result = promote_model_to_production("test_model")

        assert result is True
        mock_transition.assert_called_once_with("test_model", 2, "Production")

    @patch("src.mlflow_utils.MlflowClient")
    def test_returns_false_if_no_staging_models(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_latest_versions.return_value = []

        result = promote_model_to_production("test_model")

        assert result is False


class TestArchiveOldProductionModels:
    """Tests for archive_old_production_models function."""

    @patch("src.mlflow_utils.transition_model_stage")
    @patch("src.mlflow_utils.MlflowClient")
    def test_archives_old_versions(self, mock_client_class, mock_transition):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create 4 production versions
        versions = [MagicMock() for _ in range(4)]
        for i, v in enumerate(versions):
            v.version = i + 1
        # Sort by version descending (latest first)
        versions.reverse()
        mock_client.get_latest_versions.return_value = versions

        archive_old_production_models("test_model", keep=2)

        # Should archive versions 1 and 2 (keep 3 and 4)
        assert mock_transition.call_count == 2
        mock_transition.assert_any_call("test_model", 1, "Archived")
        mock_transition.assert_any_call("test_model", 2, "Archived")

    @patch("src.mlflow_utils.MlflowClient")
    def test_noop_if_fewer_than_keep(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        versions = [MagicMock() for _ in range(2)]
        versions[0].version = 1
        versions[1].version = 2
        mock_client.get_latest_versions.return_value = versions

        archive_old_production_models("test_model", keep=3)

        # Should not archive anything
        # (transition_model_stage not called)


class TestConstants:
    """Tests for module constants."""

    def test_experiment_name(self):
        assert EXPERIMENT_NAME == "solar-mlops-mp096"

    def test_registered_model_name(self):
        assert REGISTERED_MODEL_NAME == "solar-power-predictor"

    def test_tracking_uri(self):
        assert MLFLOW_TRACKING_URI == "sqlite:///mlflow.db"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
