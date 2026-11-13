"""Tourist Trip Design Problem (TTDP) solver package using Genetic Algorithm."""

from .models import POI, TripConfig, VisitDetail, DayRoute, ItineraryResult
from .decoder import (
    time_str_to_minutes,
    minutes_to_time_str,
    haversine_km,
    build_matrices,
    decode_chromosome,
    load_pois,
)
from .ga import GAConfig, GAResult, run_ga
from .baseline import run_greedy_baseline

__all__ = [
    "POI",
    "TripConfig",
    "VisitDetail",
    "DayRoute",
    "ItineraryResult",
    "time_str_to_minutes",
    "minutes_to_time_str",
    "haversine_km",
    "build_matrices",
    "decode_chromosome",
    "load_pois",
    "GAConfig",
    "GAResult",
    "run_ga",
    "run_greedy_baseline",
]

