import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import threading

REGISTRY_FILE = Path("models/model_registry.json")
REGISTRY_LOCK = threading.Lock()

class ModelRegistry:
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path) if registry_path else REGISTRY_FILE
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            with open(self.registry_path, 'w') as f:
                json.dump([], f)

    def _load(self) -> List[Dict[str, Any]]:
        with REGISTRY_LOCK:
            with open(self.registry_path, 'r') as f:
                return json.load(f)

    def _save(self, data: List[Dict[str, Any]]):
        with REGISTRY_LOCK:
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

    def register_model(self, metadata: Dict[str, Any]) -> None:
        data = self._load()
        metadata['registered_at'] = datetime.now().isoformat()
        data.append(metadata)
        self._save(data)

    def list_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        data = self._load()
        if not filters:
            return data
        result = []
        for model in data:
            if all(model.get(k) == v for k, v in filters.items()):
                result.append(model)
        return result

    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        data = self._load()
        updated = False
        for model in data:
            if model.get('model_id') == model_id:
                model.update(updates)
                updated = True
        if updated:
            self._save(data)
        return updated

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        data = self._load()
        for model in data:
            if model.get('model_id') == model_id:
                return model
        return None

    def delete_model(self, model_id: str) -> bool:
        data = self._load()
        new_data = [m for m in data if m.get('model_id') != model_id]
        if len(new_data) != len(data):
            self._save(new_data)
            return True
        return False 