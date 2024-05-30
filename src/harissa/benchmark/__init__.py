from harissa.benchmark.benchmark import Benchmark
from harissa.benchmark.generators import (
    NetworksGenerator as _NetworksGenerator,
    InferencesGenerator as _InferencesGenerator
)

available_networks = _NetworksGenerator.available_networks
available_inferences = _InferencesGenerator.available_inferences

__all__ = ['Benchmark', 'available_networks', 'available_inferences']