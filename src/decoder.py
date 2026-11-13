"""Distance/time matrix computation and chromosome decoder for TTDP."""

import math
from typing import Sequence

import numpy as np
import pandas as pd

from .models import DayRoute, ItineraryResult, POI, TripConfig, VisitDetail


def time_str_to_minutes(time_str: str) -> int:
    """Convert HH:MM string to minutes since 00:00."""
    hours, minutes = map(int, time_str.strip().split(":"))
    return hours * 60 + minutes


def minutes_to_time_str(minutes: float) -> str:
    """Convert minutes since 00:00 to HH:MM string."""
    total_min = int(round(minutes))
    hrs, mins = divmod(total_min, 60)
    return f"{hrs:02d}:{mins:02d}"


def load_pois(csv_path: str) -> list[POI]:
    """Load POI dataset from CSV sorted by id."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("id").reset_index(drop=True)
    pois: list[POI] = []
    for row in df.itertuples(index=False):
        pois.append(
            POI(
                id=int(row.id),
                name=str(row.name),
                lat=float(row.lat),
                lon=float(row.lon),
                rating=float(row.rating),
                duration_min=int(row.duration_min),
                open_hour=str(row.open_hour),
                close_hour=str(row.close_hour),
                ticket_cost=float(row.ticket_cost),
            )
        )
    return pois


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two lat/lon coordinates in kilometers."""
    earth_radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_km * c


def build_matrices(
    pois: Sequence[POI], speed_kmh: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build pairwise distance (km) and travel time (minutes) matrices."""
    n = len(pois)
    dist_matrix = np.zeros((n, n), dtype=float)
    time_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d_km = haversine_km(pois[i].lat, pois[i].lon, pois[j].lat, pois[j].lon)
            t_min = (d_km / speed_kmh) * 60.0
            dist_matrix[i, j] = d_km
            dist_matrix[j, i] = d_km
            time_matrix[i, j] = t_min
            time_matrix[j, i] = t_min

    return dist_matrix, time_matrix


def try_schedule_visit(
    curr_id: int,
    curr_time: float,
    poi: POI,
    committed_cost: float,
    dist_matrix: np.ndarray,
    time_matrix: np.ndarray,
    config: TripConfig,
) -> VisitDetail | None:
    """
    Check if visiting `poi` next is feasible under time-window, daily-hour,
    and total-budget constraints (including returning to the hotel).
    Returns VisitDetail if feasible, otherwise None.
    """
    travel_to = float(time_matrix[curr_id, poi.id])
    dist_to = float(dist_matrix[curr_id, poi.id])
    arrival_time = curr_time + travel_to
    wait_time = max(0.0, float(poi.open_min) - arrival_time)
    start_visit = arrival_time + wait_time
    departure_time = start_visit + float(poi.duration_min)

    travel_back = float(time_matrix[poi.id, 0])
    dist_back = float(dist_matrix[poi.id, 0])
    return_hotel_time = departure_time + travel_back

    if departure_time > float(poi.close_min) + 1e-9:
        return None
    if return_hotel_time > float(config.end_min) + 1e-9:
        return None

    leg_transport = dist_to * config.transport_cost_per_km
    return_transport = dist_back * config.transport_cost_per_km
    projected_total_cost = (
        committed_cost + leg_transport + poi.ticket_cost + return_transport
    )
    if projected_total_cost > config.b_max + 1e-9:
        return None

    return VisitDetail(
        poi_id=poi.id,
        name=poi.name,
        arrival_min=arrival_time,
        wait_min=wait_time,
        start_visit_min=start_visit,
        departure_min=departure_time,
        travel_time_min=travel_to,
        travel_distance_km=dist_to,
        ticket_cost=poi.ticket_cost,
        transport_cost=leg_transport,
    )


def decode_chromosome(
    chromosome: Sequence[int],
    pois: Sequence[POI],
    dist_matrix: np.ndarray,
    time_matrix: np.ndarray,
    config: TripConfig,
) -> ItineraryResult:
    """
    Decode a permutation of POI IDs into a feasible multi-day itinerary.

    Each day starts and ends at Hotel (id=0). Visits obey opening/closing hours,
    daily end time, total budget B_max, and at most one visit per POI.
    """
    visited: set[int] = set()
    days: list[DayRoute] = []
    visited_order: list[int] = []

    committed_cost = 0.0
    total_rating = 0.0
    total_travel_min = 0.0
    total_distance_km = 0.0

    start_min = float(config.start_min)
    end_min = float(config.end_min)

    for day_idx in range(1, config.num_days + 1):
        curr_id = 0
        curr_time = start_min
        day_route = DayRoute(day=day_idx, end_at_hotel_min=start_min)

        for poi_id in chromosome:
            if poi_id == 0 or poi_id in visited:
                continue

            poi = pois[poi_id]
            visit = try_schedule_visit(
                curr_id=curr_id,
                curr_time=curr_time,
                poi=poi,
                committed_cost=committed_cost,
                dist_matrix=dist_matrix,
                time_matrix=time_matrix,
                config=config,
            )
            if visit is None:
                continue

            # Accept visit
            visited.add(poi_id)
            visited_order.append(poi_id)
            committed_cost += visit.transport_cost + visit.ticket_cost

            day_route.visits.append(visit)
            day_route.day_rating += poi.rating
            day_route.day_travel_min += visit.travel_time_min
            day_route.day_distance_km += visit.travel_distance_km
            day_route.day_cost += visit.transport_cost + visit.ticket_cost

            curr_id = poi_id
            curr_time = visit.departure_min

        # Return to hotel at end of day
        if curr_id != 0:
            ret_time = float(time_matrix[curr_id, 0])
            ret_dist = float(dist_matrix[curr_id, 0])
            ret_cost = ret_dist * config.transport_cost_per_km

            committed_cost += ret_cost
            day_route.return_travel_min = ret_time
            day_route.return_distance_km = ret_dist
            day_route.return_transport_cost = ret_cost
            day_route.end_at_hotel_min = curr_time + ret_time
            day_route.day_travel_min += ret_time
            day_route.day_distance_km += ret_dist
            day_route.day_cost += ret_cost

        total_rating += day_route.day_rating
        total_travel_min += day_route.day_travel_min
        total_distance_km += day_route.day_distance_km
        days.append(day_route)

    fitness = total_rating - config.travel_weight * total_travel_min

    return ItineraryResult(
        days=days,
        visited_ids=visited_order,
        total_rating=total_rating,
        total_travel_minutes=total_travel_min,
        total_distance_km=total_distance_km,
        total_cost=committed_cost,
        fitness=fitness,
    )
