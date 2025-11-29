"""Preference data collection infrastructure."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict, field

from kolmogorov.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreferenceRecord:
    """A single preference comparison record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: str = ""
    
    prompt: str = ""
    chosen: str = ""
    rejected: str = ""
    
    # Metadata
    task_category: str = ""
    domain: str = ""
    conversation_depth: int = 0
    
    # Quality signals
    user_confidence: float | None = None
    response_time_ms: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    def to_dpo_format(self) -> dict[str, str]:
        """Convert to DPO trainer format."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


class PreferenceCollector:
    """Collects and stores preference data for RLHF training."""
    
    def __init__(self, storage_path: str | Path = "data/preferences"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._buffer: list[PreferenceRecord] = []
        self._buffer_size = 100
    
    def add_comparison(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        preferred: str,  # "a" or "b"
        session_id: str = "",
        **metadata,
    ) -> PreferenceRecord:
        """Add a pairwise comparison to the collection."""
        if preferred not in ("a", "b"):
            raise ValueError("preferred must be 'a' or 'b'")
        
        chosen = response_a if preferred == "a" else response_b
        rejected = response_b if preferred == "a" else response_a
        
        record = PreferenceRecord(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            session_id=session_id,
            task_category=metadata.get("task_category", ""),
            domain=metadata.get("domain", ""),
            conversation_depth=metadata.get("conversation_depth", 0),
            user_confidence=metadata.get("user_confidence"),
            response_time_ms=metadata.get("response_time_ms"),
        )
        
        self._buffer.append(record)
        
        if len(self._buffer) >= self._buffer_size:
            self.flush()
        
        return record
    
    def add_ranking(
        self,
        prompt: str,
        responses: list[str],
        ranking: list[int],  # Indices in preference order (best first)
        **metadata,
    ) -> list[PreferenceRecord]:
        """Add a ranking of multiple responses, converting to pairwise comparisons."""
        records = []
        
        for i, better_idx in enumerate(ranking[:-1]):
            for worse_idx in ranking[i + 1:]:
                record = PreferenceRecord(
                    prompt=prompt,
                    chosen=responses[better_idx],
                    rejected=responses[worse_idx],
                    **metadata,
                )
                self._buffer.append(record)
                records.append(record)
        
        if len(self._buffer) >= self._buffer_size:
            self.flush()
        
        return records
    
    def flush(self) -> None:
        """Write buffered records to storage."""
        if not self._buffer:
            return
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.storage_path / f"preferences_{timestamp}.jsonl"
        
        with open(filename, "a") as f:
            for record in self._buffer:
                f.write(json.dumps(record.to_dict()) + "\n")
        
        logger.info(f"Flushed {len(self._buffer)} records to {filename}")
        self._buffer.clear()
    
    def load_all(self) -> list[PreferenceRecord]:
        """Load all stored preference records."""
        records = []
        
        for file_path in self.storage_path.glob("*.jsonl"):
            with open(file_path) as f:
                for line in f:
                    data = json.loads(line)
                    records.append(PreferenceRecord(**data))
        
        return records
    
    def export_for_training(self, output_path: str | Path | None = None) -> list[dict]:
        """Export all records in DPO training format."""
        self.flush()
        records = self.load_all()
        
        training_data = [r.to_dpo_format() for r in records]
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(training_data, f, indent=2)
        
        return training_data
    
    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        records = self.load_all()
        
        return {
            "total_records": len(records),
            "buffered_records": len(self._buffer),
            "domains": list(set(r.domain for r in records if r.domain)),
            "categories": list(set(r.task_category for r in records if r.task_category)),
        }

