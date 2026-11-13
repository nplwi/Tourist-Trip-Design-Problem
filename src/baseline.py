"""Greedy Heuristic baseline solver for the Tourist Trip Design Problem (TTDP)."""

from typing import Sequence

import numpy as np

from .decoder import try_schedule_visit
from .models import DayRoute, ItineraryResult, POI, TripConfig, VisitDetail


def run_greedy_baseline(
    pois: Sequence[POI],
    dist_matrix: np.ndarray,
    time_matrix: np.ndarray,
    config: TripConfig,
) -> ItineraryResult:
    """
    Construct a multi-day itinerary greedily by iteratively selecting the
    feasible unvisited POI with the highest rating / distance_km ratio.
    """
    visited: set[int] = set()
    days: list[DayRoute] = []
    visited_order: list[int] = []

    committed_cost = 0.0
    total_rating = 0.0
    total_travel_min = 0.0
    total_distance_km = 0.0

    start_min = float(config.start_min)
    candidate_ids = [poi.id for poi in pois if poi.id != 0]

    for day_idx in range(1, config.num_days + 1):
        curr_id = 0
        curr_time = start_min
        day_route = DayRoute(day=day_idx, end_at_hotel_min=start_min)

        while True:
            best_visit: VisitDetail | None = None
            best_score = -1.0

            for poi_id in candidate_ids:
                if poi_id in visited:
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

                ratio = poi.rating / max(visit.travel_distance_km, 1e-6)
                if ratio > best_score:
                    best_score = ratio
                    best_visit = visit

            if best_visit is None:
                break

            poi = pois[best_visit.poi_id]
            visited.add(best_visit.poi_id)
            visited_order.append(best_visit.poi_id)
            committed_cost += best_visit.transport_cost + best_visit.ticket_cost

            day_route.visits.append(best_visit)
            day_route.day_rating += poi.rating
            day_route.day_travel_min += best_visit.travel_time_min
            day_route.day_distance_km += best_visit.travel_distance_km
            day_route.day_cost += best_visit.transport_cost + best_visit.ticket_cost

            curr_id = best_visit.poi_id
            curr_time = best_visit.departure_min

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
