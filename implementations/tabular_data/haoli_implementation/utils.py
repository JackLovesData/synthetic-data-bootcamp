"""Utilities for tabular data generation and serialized training artifacts."""

import pickle
import json
from pathlib import Path
from typing import Any

import pandas as pd


def extract_data_from_pkl(file_path: str | Path) -> Any:
	"""Load and return the Python object stored in a pickle file.

	Pickle files can execute arbitrary code when loaded. Only use this function
	with files from a trusted source. Loading also reads the complete object
	into memory.
	"""
	with Path(file_path).open("rb") as file:
		return pickle.load(file)


def read_json_file(file_path: str | Path) -> Any:
	"""Read and return the JSON value stored in a file."""
	with Path(file_path).open(encoding="utf-8") as file:
		return json.load(file)


def generate_synthetic_data(seed: int) -> pd.DataFrame:
	"""Generate single-table synthetic data using a trained TabDDPM model.

	The trained model is loaded from the standard single-table results directory.
	Generated data is saved there and returned as a DataFrame.
	"""
	from logging import INFO

	from hydra import compose, initialize_config_dir
	from midst_toolkit.common.config import ClavaDDPMMatchingConfig, ClavaDDPMSamplingConfig, GeneralConfig
	from midst_toolkit.common.logger import log
	from midst_toolkit.common.random import set_all_random_seeds
	from midst_toolkit.models.clavaddpm.data_loaders import load_tables
	from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing

	single_table_dir = Path(__file__).resolve().parents[1] / "single_table"
	base_data_dir = single_table_dir / "data"
	base_output_dir = single_table_dir / "results"
	config_dir = single_table_dir / "synthesizing"

	with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
		cfg = compose(config_name="config")

	set_all_random_seeds(seed=seed)

	log(INFO, f"Checking for a pre-trained model in {base_output_dir}...")
	tables, relation_order, _ = load_tables(base_data_dir)
	assert len(relation_order) == 1 and relation_order[0][0] is None, (
		"Relation order is not configured for single-table. "
		"For multi-table synthesizing, please use the multi-table example. "
		f"Relation order: {relation_order}"
	)

	model_file_paths = {
		relation: base_output_dir / "models" / f"{relation[0]}_{relation[1]}_ckpt.pkl"
		for relation in relation_order
	}
	missing_model_paths = [path for path in model_file_paths.values() if not path.exists()]
	if missing_model_paths:
		missing_paths = ", ".join(str(path) for path in missing_model_paths)
		raise FileNotFoundError(
			f"Trained model checkpoint(s) not found: {missing_paths}. Run the training script first."
		)

	models = {}
	for relation, model_file_path in model_file_paths.items():
		with model_file_path.open("rb") as file:
			models[relation] = pickle.load(file)

	log(INFO, "Synthesizing data...")
	synthetic_tables, _, _ = clava_synthesizing(
		tables,
		relation_order,
		base_output_dir,
		models,
		GeneralConfig(
			data_dir=base_data_dir,
			test_data_dir=base_data_dir,
			exp_name="single_table_synthesizing",
			workspace_dir=base_output_dir,
			sample_prefix="",
		),
		ClavaDDPMSamplingConfig(**cfg.sampling_config),
		ClavaDDPMMatchingConfig(**cfg.matching_config),
		sample_scale=cfg.sample_scale,
	)
	log(INFO, "Data synthesized successfully.")

	table_name = relation_order[0][1]
	return synthetic_tables[table_name]
