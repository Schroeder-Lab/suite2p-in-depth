from __future__ import annotations

import logging
from pathlib import Path

from suite2p_in_depth.general import expand_path_template, load_config, setup_suite2p_logging


def test_load_config_preserves_yaml_types(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "datasets: data/datasets.csv\n"
        "registration:\n"
        "  smooth_sigma: 1.15\n"
        "  use_gpu: false\n"
        "expression: 5 / 400\n",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config == {
        "datasets": "data/datasets.csv",
        "registration": {"smooth_sigma": 1.15, "use_gpu": False},
        "expression": "5 / 400",
    }


def test_expand_path_template_filters_requested_experiments(tmp_path: Path) -> None:
    for experiment in ("1", "3"):
        (tmp_path / "mouse-a" / "2024-01-02" / experiment).mkdir(parents=True)
    template = str(tmp_path / "{Name}" / "{Date}" / "{Experiments}")

    paths = expand_path_template(template, "mouse-a", "2024-01-02", ["1", "2", "3"])

    assert paths == [
        str(tmp_path / "mouse-a" / "2024-01-02" / "1"),
        str(tmp_path / "mouse-a" / "2024-01-02" / "3"),
    ]
    assert all(isinstance(path, str) for path in paths)


def test_expand_path_template_discovers_experiment_directories(tmp_path: Path) -> None:
    parent = tmp_path / "mouse-a" / "2024-01-02"
    (parent / "experiment-b").mkdir(parents=True)
    (parent / "experiment-a").mkdir()
    (parent / "notes.txt").write_text("not an experiment", encoding="utf-8")
    template = str(tmp_path / "{Name}" / "{Date}" / "{Experiments}")

    paths = expand_path_template(template, "mouse-a", "2024-01-02", [])

    assert set(paths) == {
        str(parent / "experiment-a"),
        str(parent / "experiment-b"),
    }


def test_expand_path_template_without_experiment_placeholder_returns_string(tmp_path: Path) -> None:
    template = str(tmp_path / "{Name}" / "{Date}" / "output")

    output = expand_path_template(template, "mouse-a", "2024-01-02", [])

    assert output == str(tmp_path / "mouse-a" / "2024-01-02" / "output")
    assert isinstance(output, str)


def test_missing_template_expansion_has_one_absence_sentinel(tmp_path: Path) -> None:
    template = str(tmp_path / "{Name}" / "{Date}" / "{Experiments}")
    assert expand_path_template(template, "mouse-a", "2024-01-02", ["missing"]) is None


def test_suite2p_logging_suppresses_tqdm_refreshes_but_keeps_summaries(
    tmp_path: Path, capsys
) -> None:
    logger = logging.getLogger("suite2p")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    for handler in original_handlers:
        logger.removeHandler(handler)

    try:
        setup_suite2p_logging(tmp_path)
        logger.info("Registering 10 frames in 1 batches")
        logger.info("0%%|          | 0/1 [00:00<?, ?it/s]")
        logger.info("100%%|##########| 1/1 [00:00<00:00, 40.75it/s]")
        logger.warning("stack_plane1_chan2.tif is a low contrast image")

        terminal_output = capsys.readouterr().out
        log_output = (tmp_path / "run_correct.log").read_text(encoding="utf-8")

        for output in (terminal_output, log_output):
            assert "Registering 10 frames in 1 batches" in output
            assert "low contrast image" in output
            assert "0%|" not in output
            assert "100%|" not in output
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            logger.addHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate
