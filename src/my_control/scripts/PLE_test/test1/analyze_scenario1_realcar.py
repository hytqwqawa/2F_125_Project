#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline analysis for Scenario-1 PLE real-car logs."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def estimate_curvature(data, window_m):
    pose = (
        data.drop_duplicates("pose_stamp")
        .sort_values("pose_stamp")
        .reset_index()
    )
    x = pose["pose_x"].to_numpy(float)
    y = pose["pose_y"].to_numpy(float)
    yaw = np.unwrap(pose["yaw_corrected"].to_numpy(float))
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    curvature = np.full(len(pose), np.nan)

    half = window_m / 2.0
    for index, center in enumerate(arc):
        selected = np.abs(arc-center) <= half
        if np.count_nonzero(selected) < 9:
            continue
        local_arc = arc[selected]-center
        coefficients = np.polyfit(local_arc, yaw[selected], 2)
        curvature[index] = coefficients[1]

    output = np.full(len(data), np.nan)
    output[pose["index"].to_numpy(int)] = curvature
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--output-dir", default="scenario1_analysis")
    parser.add_argument("--curvature-window", type=float, default=0.20)
    parser.add_argument("--tau-model", type=float, default=0.105)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for filename in args.logs:
        data = pd.read_csv(filename)
        data["kappa_actual_est"] = estimate_curvature(
            data, args.curvature_window
        )

        elapsed = data["elapsed_time"].to_numpy(float)
        dt = np.r_[0.0, np.diff(elapsed)]
        kappa_command = data["kappa_cmd"].to_numpy(float)
        model = np.zeros(len(data))
        model[0] = kappa_command[0]
        for index in range(1, len(data)):
            step = min(max(dt[index], 0.0), 0.10)
            model[index] = model[index-1] + step * (
                kappa_command[index-1]-model[index-1]
            ) / max(args.tau_model, 1.0e-4)

        data["kappa_actuator_model"] = model
        data["d_kappa_cmd"] = (
            data["kappa_actual_est"]-data["kappa_cmd"]
        )
        data["r_kappa_model"] = (
            data["kappa_actual_est"]-data["kappa_actuator_model"]
        )

        ds = np.r_[0.0, np.maximum(
            np.diff(data["s"].to_numpy(float)), 0.0
        )]
        b_bar = data["b_bar_cert"].to_numpy(float)
        residual = data["r_kappa_model"].to_numpy(float)
        valid = np.isfinite(b_bar) & np.isfinite(residual)
        increment = np.zeros(len(data))
        increment[valid] = (
            b_bar[valid]**2 * residual[valid]**2 * ds[valid]
        )
        data["residual_weighted_energy"] = np.cumsum(increment)

        insertion = data["s"] >= float(data["distance_to_insertion"].iloc[0] * 0.0)
        pre_in = data["event_inserted"].to_numpy(int) == 0
        event = data["event_discovered"].to_numpy(int) == 1
        event_pre_in = event & pre_in
        final_energy = (
            float(data.loc[event_pre_in, "residual_weighted_energy"].iloc[-1])
            if np.any(event_pre_in) else float("nan")
        )

        terminal_rows = data[
            np.abs(data["distance_to_insertion"]) <= 0.05
        ]
        if terminal_rows.empty:
            terminal_e = float("nan")
            terminal_delta = float("nan")
        else:
            terminal_e = float(terminal_rows["e"].abs().max())
            terminal_delta = float(
                terminal_rows["delta_deg"].abs().max()
            )

        summaries.append({
            "file": os.path.basename(filename),
            "finished": bool(data["state"].iloc[-1] == "FINISHED"),
            "fault_code": str(data["fault_code"].iloc[-1]),
            "max_abs_e_m": float(data["e"].abs().max()),
            "max_abs_delta_deg": float(data["delta_deg"].abs().max()),
            "max_V_over_beta": float(
                data["V_gamma_over_beta"].max()
            ),
            "max_abs_kappa_cmd_1_m": float(
                data["kappa_cmd"].abs().max()
            ),
            "terminal_abs_e_m": terminal_e,
            "terminal_abs_delta_deg": terminal_delta,
            "residual_weighted_energy_to_insertion": final_energy,
            "allowed_residual_energy": 3.67479354735e-05,
            "energy_pass": bool(
                np.isfinite(final_energy)
                and final_energy <= 3.67479354735e-05
            ),
        })

        stem = Path(filename).stem
        enriched_path = output_dir / f"{stem}_enriched.csv"
        data.to_csv(enriched_path, index=False)

        figure = plt.figure(figsize=(8.2, 5.2))
        axis = figure.add_subplot(111)
        axis.plot(data["s"], data["kappa_cmd"], label="Commanded")
        axis.plot(data["s"], data["kappa_actual_est"], label="Estimated actual")
        axis.plot(data["s"], data["kappa_actuator_model"], label="First-order model")
        axis.set_xlabel("Reference coordinate s [m]")
        axis.set_ylabel("Curvature [1/m]")
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{stem}_curvature.png", dpi=180
        )
        plt.close(figure)

        figure = plt.figure(figsize=(8.2, 5.2))
        axis = figure.add_subplot(111)
        axis.plot(data["s"], data["e"], label="Lateral error")
        axis.axhline(0.05, linestyle="--", label="Nominal tube")
        axis.axhline(-0.05, linestyle="--")
        axis.axhline(0.085, linestyle=":", label="Actuator tube")
        axis.axhline(-0.085, linestyle=":")
        axis.set_xlabel("Reference coordinate s [m]")
        axis.set_ylabel("e [m]")
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{stem}_lateral_error.png", dpi=180
        )
        plt.close(figure)

        figure = plt.figure(figsize=(8.2, 5.2))
        axis = figure.add_subplot(111)
        axis.plot(
            data["s"], data["residual_weighted_energy"],
            label="Measured residual energy"
        )
        axis.axhline(
            3.67479354735e-05, linestyle="--",
            label="Current allowable energy"
        )
        axis.set_xlabel("Reference coordinate s [m]")
        axis.set_ylabel("Weighted residual energy")
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{stem}_residual_energy.png", dpi=180
        )
        plt.close(figure)

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "scenario1_run_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
