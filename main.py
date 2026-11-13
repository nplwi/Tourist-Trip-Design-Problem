"""Entry point for running TTDP experiments (Genetic Algorithm vs Greedy Baseline)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import run_greedy_baseline
from src.decoder import (
    build_matrices,
    load_pois,
    minutes_to_time_str,
)
from src.ga import GAConfig, GAResult, run_ga
from src.models import ItineraryResult, POI, TripConfig


def verify_constraints(
    result: ItineraryResult, pois: list[POI], config: TripConfig
) -> tuple[bool, list[str]]:
    """Verify that the itinerary satisfies all TTDP hard constraints."""
    errors: list[str] = []

    if len(result.visited_ids) != len(set(result.visited_ids)):
        errors.append("Duplicate POI visits detected.")

    if result.total_cost > config.b_max + 1e-6:
        errors.append(
            f"Budget exceeded: {result.total_cost:.2f} > {config.b_max:.2f}"
        )

    for day_route in result.days:
        for visit in day_route.visits:
            poi = pois[visit.poi_id]
            if visit.start_visit_min < poi.open_min - 1e-6:
                errors.append(
                    f"Day {day_route.day} {poi.name}: started before open_hour."
                )
            if visit.departure_min > poi.close_min + 1e-6:
                errors.append(
                    f"Day {day_route.day} {poi.name}: departed after close_hour."
                )
        if day_route.visits and day_route.end_at_hotel_min > config.end_min + 1e-6:
            errors.append(
                f"Day {day_route.day}: returned to hotel after {config.end_time}."
            )

    return len(errors) == 0, errors


def save_itinerary_json(
    output_path: Path,
    ga_result: GAResult,
    greedy_result: ItineraryResult,
    config: TripConfig,
    constraints_ok: bool,
) -> None:
    """Serialize the best itinerary and benchmark summary to JSON."""
    itinerary = ga_result.best_itinerary
    payload = {
        "config": {
            "num_days": config.num_days,
            "start_time": config.start_time,
            "end_time": config.end_time,
            "b_max_idr": config.b_max,
            "transport_cost_per_km_idr": config.transport_cost_per_km,
            "speed_kmh": config.speed_kmh,
            "travel_weight": config.travel_weight,
        },
        "constraints_satisfied": constraints_ok,
        "best_ga_summary": {
            "seed": ga_result.seed,
            "fitness": round(itinerary.fitness, 6),
            "total_rating": round(itinerary.total_rating, 4),
            "total_pois_visited": len(itinerary.visited_ids),
            "total_travel_minutes": round(itinerary.total_travel_minutes, 2),
            "total_distance_km": round(itinerary.total_distance_km, 2),
            "total_cost_idr": round(itinerary.total_cost, 2),
            "runtime_sec": round(ga_result.runtime_sec, 4),
        },
        "greedy_baseline_summary": {
            "fitness": round(greedy_result.fitness, 6),
            "total_rating": round(greedy_result.total_rating, 4),
            "total_pois_visited": len(greedy_result.visited_ids),
            "total_travel_minutes": round(greedy_result.total_travel_minutes, 2),
            "total_distance_km": round(greedy_result.total_distance_km, 2),
            "total_cost_idr": round(greedy_result.total_cost, 2),
        },
        "days": [],
    }

    for day_route in itinerary.days:
        day_dict = {
            "day": day_route.day,
            "start_from": "Hotel Simpang Lima (Depot)",
            "start_time": config.start_time,
            "end_at_hotel_time": minutes_to_time_str(day_route.end_at_hotel_min),
            "day_rating": round(day_route.day_rating, 2),
            "day_travel_minutes": round(day_route.day_travel_min, 2),
            "day_distance_km": round(day_route.day_distance_km, 2),
            "day_cost_idr": round(day_route.day_cost, 2),
            "visits": [
                {
                    "poi_id": v.poi_id,
                    "name": v.name,
                    "arrival_time": minutes_to_time_str(v.arrival_min),
                    "wait_minutes": round(v.wait_min, 2),
                    "start_visit_time": minutes_to_time_str(v.start_visit_min),
                    "departure_time": minutes_to_time_str(v.departure_min),
                    "travel_time_min": round(v.travel_time_min, 2),
                    "travel_distance_km": round(v.travel_distance_km, 2),
                    "ticket_cost_idr": round(v.ticket_cost, 2),
                    "transport_cost_idr": round(v.transport_cost, 2),
                }
                for v in day_route.visits
            ],
            "return_to_hotel": {
                "travel_time_min": round(day_route.return_travel_min, 2),
                "distance_km": round(day_route.return_distance_km, 2),
                "transport_cost_idr": round(day_route.return_transport_cost, 2),
            },
        }
        payload["days"].append(day_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_convergence(
    output_path: Path, ga_result: GAResult, greedy_fitness: float
) -> None:
    """Plot GA best and average fitness convergence vs Greedy baseline."""
    generations = np.arange(1, len(ga_result.best_fitness_history) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(
        generations,
        ga_result.best_fitness_history,
        label=f"GA Best Fitness (seed={ga_result.seed})",
        color="#1f77b4",
        linewidth=2.2,
    )
    plt.plot(
        generations,
        ga_result.avg_fitness_history,
        label="GA Average Fitness",
        color="#ff7f0e",
        linewidth=1.8,
        alpha=0.85,
    )
    plt.axhline(
        y=greedy_fitness,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label=f"Greedy Baseline ({greedy_fitness:.3f})",
    )

    plt.title("Konvergensi Fitness Genetic Algorithm pada TTDP Semarang")
    plt.xlabel("Generasi")
    plt.ylabel("Fitness")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_routes(
    output_path: Path, pois: list[POI], itinerary: ItineraryResult
) -> None:
    """Plot spatial coordinates of POIs and directional daily routes."""
    plt.figure(figsize=(10, 8))

    # Plot all unvisited POIs in gray
    visited_set = set(itinerary.visited_ids)
    unvisited = [p for p in pois if p.id != 0 and p.id not in visited_set]
    if unvisited:
        plt.scatter(
            [p.lon for p in unvisited],
            [p.lat for p in unvisited],
            c="#b0b0b0",
            s=60,
            alpha=0.7,
            label="POI Tidak Dikunjungi",
            zorder=2,
        )
        for p in unvisited:
            plt.annotate(
                f"{p.id}. {p.name}",
                (p.lon, p.lat),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7.5,
                color="#666666",
            )

    # Plot Hotel (Depot)
    hotel = pois[0]
    plt.scatter(
        [hotel.lon],
        [hotel.lat],
        c="#d62728",
        marker="*",
        s=260,
        edgecolors="black",
        linewidths=0.8,
        label="Hotel / Depot (ID 0)",
        zorder=5,
    )
    plt.annotate(
        "0. Hotel (Depot)",
        (hotel.lon, hotel.lat),
        textcoords="offset points",
        xytext=(7, -10),
        fontsize=8.5,
        fontweight="bold",
        color="#d62728",
    )

    day_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]

    for idx, day_route in enumerate(itinerary.days):
        if not day_route.visits:
            continue
        color = day_colors[idx % len(day_colors)]
        route_ids = [0] + [v.poi_id for v in day_route.visits] + [0]

        visited_pois_day = [pois[v.poi_id] for v in day_route.visits]
        plt.scatter(
            [p.lon for p in visited_pois_day],
            [p.lat for p in visited_pois_day],
            c=color,
            s=95,
            edgecolors="black",
            linewidths=0.6,
            label=f"Hari {day_route.day} ({len(day_route.visits)} POI)",
            zorder=4,
        )

        for p in visited_pois_day:
            plt.annotate(
                f"{p.id}. {p.name}",
                (p.lon, p.lat),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                fontweight="bold",
                color=color,
            )

        for u_id, v_id in zip(route_ids[:-1], route_ids[1:]):
            u, v = pois[u_id], pois[v_id]
            plt.annotate(
                "",
                xy=(v.lon, v.lat),
                xytext=(u.lon, u.lat),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.7,
                    shrinkA=6,
                    shrinkB=6,
                    alpha=0.85,
                ),
                zorder=3,
            )

    plt.title("Visualisasi Rute Wisata Harian Terbaik (GA - Semarang)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_itinerary_details(itinerary: ItineraryResult, config: TripConfig) -> None:
    """Print detailed day-by-day itinerary to stdout."""
    for day_route in itinerary.days:
        print(
            f"\n--- Hari {day_route.day} | Rating: {day_route.day_rating:.1f} | "
            f"Jarak: {day_route.day_distance_km:.2f} km | "
            f"Travel: {day_route.day_travel_min:.1f} mnt | "
            f"Biaya: Rp {day_route.day_cost:,.0f} ---"
        )
        print(f"  Berangkat dari Hotel (ID 0) pukul {config.start_time}")
        for idx, v in enumerate(day_route.visits, start=1):
            wait_info = f" (tunggu {v.wait_min:.0f} mnt)" if v.wait_min > 0 else ""
            print(
                f"  {idx}. [{v.poi_id:2d}] {v.name:<34} | "
                f"Tiba {minutes_to_time_str(v.arrival_min)}{wait_info} | "
                f"Kunjungan {minutes_to_time_str(v.start_visit_min)}-{minutes_to_time_str(v.departure_min)} | "
                f"Tiket Rp {v.ticket_cost:,.0f}"
            )
        if day_route.visits:
            print(
                f"  Kembali ke Hotel (ID 0) pukul {minutes_to_time_str(day_route.end_at_hotel_min)} "
                f"(+{day_route.return_travel_min:.1f} mnt, {day_route.return_distance_km:.2f} km)"
            )


def main() -> None:
    """Run GA experiments, compare with Greedy baseline, and save outputs."""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "poi_semarang.csv"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pois = load_pois(str(data_path))
    trip_config = TripConfig(
        num_days=2,
        start_time="08:00",
        end_time="18:00",
        b_max=350000.0,
        transport_cost_per_km=3000.0,
        speed_kmh=30.0,
        travel_weight=0.001,
    )
    ga_config = GAConfig(
        population_size=80,
        generations=150,
        crossover_rate=0.85,
        mutation_rate=0.25,
        tournament_size=3,
        elitism_count=2,
    )

    dist_matrix, time_matrix = build_matrices(pois, speed_kmh=trip_config.speed_kmh)

    # 1. Greedy Baseline
    greedy_result = run_greedy_baseline(pois, dist_matrix, time_matrix, trip_config)

    # 2. Genetic Algorithm (10 independent runs)
    seeds = list(range(42, 52))
    ga_runs: list[GAResult] = []
    for seed in seeds:
        res = run_ga(
            pois=pois,
            dist_matrix=dist_matrix,
            time_matrix=time_matrix,
            trip_config=trip_config,
            ga_config=ga_config,
            seed=seed,
        )
        ga_runs.append(res)

    fitness_values = np.array([r.best_itinerary.fitness for r in ga_runs])
    runtimes = np.array([r.runtime_sec for r in ga_runs])

    best_ga_run = max(ga_runs, key=lambda r: r.best_itinerary.fitness)
    best_itinerary = best_ga_run.best_itinerary

    constraints_ok, constraint_errors = verify_constraints(
        best_itinerary, pois, trip_config
    )
    greedy_ok, greedy_errors = verify_constraints(greedy_result, pois, trip_config)

    improvement_pct = (
        (best_itinerary.fitness - greedy_result.fitness) / greedy_result.fitness
    ) * 100.0

    print("=" * 72)
    print("TOURIST TRIP DESIGN PROBLEM (TTDP) - SEMARANG (2 HARI)")
    print("=" * 72)
    print(f"Jumlah POI (non-depot) : {len(pois) - 1}")
    print(f"Anggaran Maksimal      : Rp {trip_config.b_max:,.0f}")
    print(f"Jam Operasional Harian : {trip_config.start_time} - {trip_config.end_time}")
    print("-" * 72)
    print(
        f"Greedy Baseline        | Fitness: {greedy_result.fitness:.4f} | "
        f"Rating: {greedy_result.total_rating:.1f} | "
        f"POI: {len(greedy_result.visited_ids)} | "
        f"Travel: {greedy_result.total_travel_minutes:.1f} mnt | "
        f"Biaya: Rp {greedy_result.total_cost:,.0f} | Valid: {greedy_ok}"
    )
    print(
        f"GA Best (seed={best_ga_run.seed})      | Fitness: {best_itinerary.fitness:.4f} | "
        f"Rating: {best_itinerary.total_rating:.1f} | "
        f"POI: {len(best_itinerary.visited_ids)} | "
        f"Travel: {best_itinerary.total_travel_minutes:.1f} mnt | "
        f"Biaya: Rp {best_itinerary.total_cost:,.0f} | Valid: {constraints_ok}"
    )
    print("-" * 72)
    print(f"GA 10 Runs Mean Fitness: {fitness_values.mean():.4f} +/- {fitness_values.std():.4f}")
    print(f"GA 10 Runs Avg Runtime : {runtimes.mean():.4f} detik")
    print(f"Peningkatan GA vs Greedy: +{improvement_pct:.2f}%")
    print("=" * 72)

    if not constraints_ok:
        print("PERINGATAN PELANGGARAN KENDALA:", constraint_errors)
    if not greedy_ok:
        print("PERINGATAN GREEDY:", greedy_errors)

    print_itinerary_details(best_itinerary, trip_config)

    # Save artifacts
    json_path = results_dir / "best_itinerary.json"
    conv_path = results_dir / "convergence_plot.png"
    route_path = results_dir / "route_plot.png"

    save_itinerary_json(
        json_path, best_ga_run, greedy_result, trip_config, constraints_ok
    )
    plot_convergence(conv_path, best_ga_run, greedy_result.fitness)
    plot_routes(route_path, pois, best_itinerary)

    print(f"\nOutput disimpan ke:")
    print(f"  - {json_path}")
    print(f"  - {conv_path}")
    print(f"  - {route_path}")


if __name__ == "__main__":
    main()

