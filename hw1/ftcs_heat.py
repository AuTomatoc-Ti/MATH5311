#!/usr/bin/env python3
"""
FTCS solver for 1D heat equation on [0, pi] with manufactured exact solution.

PDE: u_t = u_xx + f(x,t),  0 < x < pi, 0 < t <= T
BC:  u(0,t) = 0, u(pi,t) = 0
IC:  u(x,0) = sin(x)

Manufactured exact solution: u(x,t) = sin(x) cos(t).
Then u_t = -sin(x) sin(t), u_xx = -sin(x) cos(t) so
f(x,t) = u_t - u_xx = sin(x) (cos(t) - sin(t)).

Time step choice: Let M = ceil(T / (0.5 * dx^2)), dt = T / M (so dt <= 0.5 dx^2)

We report the max-norm error at T: max_j |u_j^M - u(x_j, T)| and plot vs dx on log-log scale
for N in {20, 40, 80, 160}.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class RunResult:
    N: int
    dx: float
    M: int
    dt: float
    err_max: float


def u_exact(x: np.ndarray | float, t: float) -> np.ndarray | float:
    return np.sin(x) * np.cos(t)


def f_source(x: np.ndarray | float, t: float) -> np.ndarray | float:
    # For the chosen exact solution, f(x,t) = sin(x) * (cos t - sin t)
    return np.sin(x) * (np.cos(t) - np.sin(t))


def solve_heat_ftcs(N: int, T: float, f: Callable[[np.ndarray | float, float], np.ndarray | float]) -> RunResult:
    """
    Solve u_t = u_xx + f(x,t) with FTCS on [0, pi] x [0, T].

    Grid: x_j = j*dx, j=0..N, dx = pi/N. Dirichlet BC: u(0,t)=u(pi,t)=0.
    Time: M = ceil(T/(0.5*dx^2)), dt = T/M.

    Returns RunResult with max error at T against u_exact.
    """
    L = math.pi
    dx = L / N
    M = math.ceil(T / (0.5 * dx * dx))
    dt = T / M
    r = dt / (dx * dx)  # stability requires r <= 1/2 (satisfied by construction)

    # spatial grid
    x = np.linspace(0.0, L, N + 1)

    # initial condition: u(x,0) = u_exact(x, 0) = sin(x)
    u = u_exact(x, 0.0).astype(float)

    # enforce Dirichlet BCs at boundaries for all times
    u[0] = 0.0
    u[-1] = 0.0

    # time stepping
    for n in range(M):
        t_n = n * dt
        # compute source at time level n (explicit)
        f_n = f(x, t_n)
        # new array to store u^{n+1}
        u_new = u.copy()
        # FTCS update for interior points
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2]) + dt * f_n[1:-1]
        # boundaries remain zero
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new

    # compute error at final time T
    u_T_exact = u_exact(x, T)
    err_max = float(np.max(np.abs(u - u_T_exact)))

    return RunResult(N=N, dx=dx, M=M, dt=dt, err_max=err_max)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FTCS heat equation convergence study")
    parser.add_argument("--Ns", type=str, default="20,40,80,160", help="Comma-separated N values, e.g. '20,40,80,160'")
    parser.add_argument("--T", type=float, default=1.0, help="Final time T")
    parser.add_argument("--out", type=str, default="hw1/error_plot_ftcs.png", help="Output path for saved plot")
    args = parser.parse_args()

    Ns = [int(s) for s in args.Ns.split(",")]
    T = args.T

    results: List[RunResult] = []
    for N in Ns:
        res = solve_heat_ftcs(N, T, f_source)
        results.append(res)
        print(f"N={res.N:4d}  dx={res.dx:.6e}  M={res.M:7d}  dt={res.dt:.6e}  err_max={res.err_max:.6e}")

    # sort by dx ascending just in case
    results.sort(key=lambda r: r.dx)

    dxs = np.array([r.dx for r in results])
    errs = np.array([r.err_max for r in results])

    # Create log-log plot
    plt.figure(figsize=(6, 4))
    plt.loglog(dxs, errs, "o-", label="FTCS error")

    # reference slope-2 line (c * dx^2) scaled to pass through the last point
    c = errs[-1] / (dxs[-1] ** 2)
    plt.loglog(dxs, c * dxs ** 2, "k--", label="O(dx^2)")

    plt.xlabel("dx = pi/N")
    plt.ylabel("max_j |u_j^M - u(x_j,T)|")
    plt.title("FTCS convergence: heat eq. on [0, pi]")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # ensure output directory exists (relative to repo root)
    import os
    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_path, dpi=150)
    print(f"Saved log-log error plot to: {out_path}")


if __name__ == "__main__":
    main()
