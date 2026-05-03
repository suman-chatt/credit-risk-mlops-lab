from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    age_min: int = 1
    debt_ratio_cap: float = 10.0
    revolving_utilization_cap: float = 2.0