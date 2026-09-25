import argparse
import copy
import importlib
import logging
import os
from datetime import datetime

import pandas as pd
from suite2p import parameters

from . import general, zregister

# Import the *module* `suite2p.run_s2p` (avoids collision with `from suite2p import run_s2p`)
run_s2p_mod = importlib.import_module("suite2p.run_s2p")

logger = logging.getLogger("suite2p")


def parse_args() -> argparse.Namespace:
    """Parse the configuration path for the standalone Suite2p runner.

    Returns
    -------
    argparse.Namespace
        Parsed ``config`` path, defaulting to ``plain_suite2p.yaml``.
    """
    parser = argparse.ArgumentParser(description="Run suite2p")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="plain_suite2p.yaml",
        help="Path to configuration file (yaml)",
    )
    return parser.parse_args()


def process_all_datasets(config: dict, datasets: pd.DataFrame) -> None:
    """Run Suite2p on each selected dataset with configured settings.

    Parameters
    ----------
    config : dict
        Configuration containing directory templates and Suite2p settings.
    datasets : pandas.DataFrame
        Dataset rows with ``Name``, ``Date``, and ``Process`` columns.
        Rows without imaging TIFFs are skipped.
    """
    for i in range(len(datasets)):
        if not datasets.loc[i]["Process"]:
            continue
        logger.info("Starting dataset %s/%s", datasets.loc[i]["Name"], datasets.loc[i]["Date"])
        paths = zregister.make_paths(datasets.loc[i], config["directories"])

        run_s2p_mod.logger_setup(os.path.join(paths["output"], "suite2p"))

        logger.info(f"Processing {datasets.loc[i]['Name']} {datasets.loc[i]['Date']}...")
        if paths["tiffs"] is None:
            logger.info(
                f"NOTE: Imaging directory for {datasets.loc[i]['Name']} "
                f"{datasets.loc[i]['Date']} not found. Skipping.\n\n"
            )
            continue

        # Create settings and db (as used in suite2p).
        settings = parameters.default_settings()
        user_settings = zregister.get_settings(datasets.loc[i], config["suite2p_settings"])
        user_settings["date_proc"] = datetime.now().astimezone()
        parameters.set_settings(settings, copy.deepcopy(user_settings))

        db = parameters.default_db()
        parameters.set_db(db, zregister.get_db(paths, datasets.loc[i], copy.deepcopy(config)))

        run_s2p_mod.run_s2p(db=db, settings=settings)


def main() -> None:
    """Load the standalone runner configuration and process its datasets.

    Missing configuration or dataset files terminate with a nonzero status.
    """
    args = parse_args()
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        exit(1)
    conf = general.load_config(args.config)
    if not os.path.exists(conf["datasets"]):
        print(f"Dataset file {conf['datasets']} does not exist.")
        exit(1)
    ds = zregister.load_dataset_list(conf["datasets"])
    if ds is None:
        exit(1)

    process_all_datasets(conf, ds)


if __name__ == "__main__":
    main()
