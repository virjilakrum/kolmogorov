"""Preference dataset handling for RLHF training."""

from typing import Any
from pathlib import Path

from datasets import Dataset, load_dataset, DatasetDict

from kolmogorov.utils.logging import get_logger

logger = get_logger(__name__)


class PreferenceDataset:
    """Handles loading and formatting preference datasets for various trainers."""
    
    REQUIRED_DPO_COLUMNS = {"prompt", "chosen", "rejected"}
    REQUIRED_REWARD_COLUMNS = {"chosen", "rejected"}
    
    def __init__(
        self,
        dataset_name_or_path: str | None = None,
        data: list[dict[str, Any]] | None = None,
        split: str = "train",
    ):
        self.dataset_name = dataset_name_or_path
        self._dataset: Dataset | None = None
        self._split = split
        
        if data is not None:
            self._dataset = Dataset.from_list(data)
        elif dataset_name_or_path:
            self._load_dataset(dataset_name_or_path, split)
    
    def _load_dataset(self, name_or_path: str, split: str) -> None:
        """Load dataset from HuggingFace Hub or local path."""
        path = Path(name_or_path)
        
        if path.exists():
            if path.suffix == ".json":
                self._dataset = Dataset.from_json(str(path))
            elif path.suffix == ".jsonl":
                self._dataset = Dataset.from_json(str(path))
            elif path.is_dir():
                self._dataset = load_dataset("json", data_dir=str(path), split=split)
        else:
            logger.info(f"Loading dataset from HuggingFace: {name_or_path}")
            ds = load_dataset(name_or_path, split=split)
            self._dataset = ds if isinstance(ds, Dataset) else ds[split]
    
    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            raise ValueError("No dataset loaded")
        return self._dataset
    
    def validate_for_dpo(self) -> bool:
        """Check if dataset has required columns for DPO training."""
        columns = set(self.dataset.column_names)
        has_prompt = "prompt" in columns
        has_chosen = "chosen" in columns
        has_rejected = "rejected" in columns
        
        if not (has_chosen and has_rejected):
            logger.error(f"Missing required columns. Found: {columns}")
            return False
        
        if not has_prompt:
            logger.warning("No 'prompt' column found. Using implicit prompt format.")
        
        return True
    
    def validate_for_reward(self) -> bool:
        """Check if dataset has required columns for reward model training."""
        columns = set(self.dataset.column_names)
        return "chosen" in columns and "rejected" in columns
    
    def get_train_dataset(self) -> Dataset:
        """Get the training dataset."""
        return self.dataset
    
    def filter_quality(self, min_length: int = 10) -> "PreferenceDataset":
        """Filter out low-quality examples."""
        def has_min_length(example: dict) -> bool:
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")
            
            if isinstance(chosen, list):
                chosen = chosen[-1].get("content", "") if chosen else ""
            if isinstance(rejected, list):
                rejected = rejected[-1].get("content", "") if rejected else ""
            
            return len(str(chosen)) >= min_length and len(str(rejected)) >= min_length
        
        filtered = self.dataset.filter(has_min_length)
        logger.info(f"Filtered dataset: {len(self.dataset)} -> {len(filtered)} examples")
        
        new_instance = PreferenceDataset()
        new_instance._dataset = filtered
        return new_instance
    
    def __len__(self) -> int:
        return len(self.dataset) if self._dataset else 0
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[idx]

