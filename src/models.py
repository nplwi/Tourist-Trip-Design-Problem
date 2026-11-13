"""Data models for Tourist Trip Design Problem (TTDP)."""

from dataclasses import dataclass, field


def _parse_hhmm(time_str: str) -> int:
    """Convert HH:MM string into minutes since 00:00."""
    hours, minutes = map(int, time_str.strip().split(":"))
    return hours * 60 + minutes


@dataclass(frozen=True)
class POI:
    """Point of Interest (or Hotel Depot when id == 0)."""

    id: int
    name: str
    lat: float
    lon: float
    rating: float
    duration_min: int
    open_hour: str
    close_hour: str
    ticket_cost: float

    @property
    def open_min(self) -> int:
        """Opening time in minutes since midnight."""
        return _parse_hhmm(self.open_hour)

    @property
    def close_min(self) -> int:
        """Closing time in minutes since midnight."""
        return _parse_hhmm(self.close_hour)


@dataclass(frozen=True)
class TripConfig:
    """Trip configuration and constraint parameters."""

    num_days: int = 2
    start_time: str = "08:00"
    end_time: str = "18:00"
    b_max: float = 350000.0
    transport_cost_per_km: float = 3000.0
    speed_kmh: float = 30.0
    travel_weight: float = 0.001

    @property
    def start_min(self) -> int:
        """Daily start time in minutes since midnight."""
        return _parse_hhmm(self.start_time)

    @property
    def end_min(self) -> int:
        """Daily end time in minutes since midnight."""
        return _parse_hhmm(self.end_time)


@dataclass
class VisitDetail:
    """Schedule and cost details for a single visited POI."""

    poi_id: int
    name: str
    arrival_min: float
    wait_min: float
    start_visit_min: float
    departure_min: float
    travel_time_min: float
    travel_distance_km: float
    ticket_cost: float
    transport_cost: float


@dataclass
class DayRoute:
    """Complete itinerary for a single day starting and ending at the Hotel."""

    day: int
    visits: list[VisitDetail] = field(default_factory=list)
    return_travel_min: float = 0.0
    return_distance_km: float = 0.0
    return_transport_cost: float = 0.0
    end_at_hotel_min: float = 0.0
    day_rating: float = 0.0
    day_travel_min: float = 0.0
    day_distance_km: float = 0.0
    day_cost: float = 0.0


@dataclass
class ItineraryResult:
    """Decoded multi-day itinerary and objective evaluation."""

    days: list[DayRoute]
    visited_ids: list[int]
    total_rating: float
    total_travel_minutes: float
    total_distance_km: float
    total_cost: float
    fitness: float

