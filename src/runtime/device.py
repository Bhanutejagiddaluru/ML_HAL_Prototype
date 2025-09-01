
from dataclasses import dataclass

@dataclass
class Device:
    name: str  # 'cpu', 'mock_accel'
    id: int = 0

def current_device() -> Device:
    # could consult env vars, availability, etc.
    return Device('mock_accel', 0)
