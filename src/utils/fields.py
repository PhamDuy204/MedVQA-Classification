from typing import Any, Dict, Iterable


def get_field(sample: Dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in sample and sample[name] is not None:
            return sample[name]
    return default
