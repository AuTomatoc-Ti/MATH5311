#!/usr/bin/env python3
"""
Implicit (Backward Euler) solver for 1D heat equation on [0, pi] with manufactured exact solution.

PDE: u_t = u_xx + f(x,t),  0 < x < pi, 0 < t <= T
BC:  u(0,t) = 0, u(pi,t) = 0
IC:  u(x,0) = sin(x)

Manufactured exact solution: u(x,t) = sin(x) cos(t).
Then u_t = -sin(x) sin(t), u_xx = -sin(x) cos(t) so
f(x,t) = u_t - u_xx = sin(x) (cos(t) - sin(t)).

Time step choice: Let M = ceil(T / (0.5 * dx^2)), dt = T / M (mirroring the explicit rule used in the prompt).

We report the max-norm error at T: max_j |u_j^M - u(x_j, T)| and plot vs dx on log-log scale
for N in {20, 40, 80, 160}.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class RunResult:
    N: int
    dx: float
    M: int
    dt: float
    err_max: float


def u_exact(x: np.ndarray | float, t: float):
    return np.sin(x) * np.cos(t)


def f_source(x: np.ndarray | float, t: float):
    return np.sin(x) * (np.cos(t) - np.sin(t))


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve tridiagonal system Ax = d using Thomas algorithm.
    a: sub-diagonal (length m-1)
    b: diagonal (length m)
    c: super-diagonal (length m-1)
    Returns x (length m)
    """
    m = len(b)
    ac, bc, cc, dc = map(np.array, (a.copy(), b.copy(), c.copy(), d.copy()))
    # Forward sweep
    for i in range(1, m):
        w = ac[i-1] / bc[i-1]
        bc[i] = bc[i] - w * cc[i-1]
        dc[i] = dc[i] - w * dc[i-1]
    # Back substitution
    x = np.zeros_like(dc)
    x[-1] = dc[-1] / bc[-1]
    for i in range(m - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


def solve_heat_backward_euler(N: int, T: float) -> RunResult:
    L = math.pi
    dx = L / N
    M = math.ceil(T / (0.5 * dx * dx))
    dt = T / M
    r = dt / (dx * dx)

    # grid
    x = np.linspace(0.0, L, N + 1)

    # initial condition
    u = u_exact(x, 0.0).astype(float)
    u[0] = 0.0
    u[-1] = 0.0

    # Build constant tridiagonal for interior unknowns (size m=N-1)
    m = N - 1
    # matrix for (I - dt * D_xx)
    # D_xx discretization gives -2/h^2 on diag and 1/h^2 on off-diagonals,
    # so A = I - dt * D_xx leads to
    # diag = 1 + 2r, off-diag = -r
    a = -r * np.ones(m - 1)
    b = (1 + 2 * r) * np.ones(m)
    c = -r * np.ones(m - 1)

    for n in range(M):
        t_np1 = (n + 1) * dt
        # RHS: u^n + dt * f^{n+1} on interior
        rhs = u[1:-1] + dt * f_source(x[1:-1], t_np1)
        # Dirichlet BC contribution (both boundaries are zero -> no addition)
        # Solve tridiagonal system for interior unknowns
        u_interior = thomas_solve(a, b, c, rhs)
        u[1:-1] = u_interior
        u[0] = 0.0
        u[-1] = 0.0

    # error at T
    err = float(np.max(np.abs(u - u_exact(x, T))))
    return RunResult(N=N, dx=dx, M=M, dt=dt, err_max=err)


def main():
    import argparse, os
    parser = argparse.ArgumentParser(description="Implicit (Backward Euler) heat equation convergence study")
    parser.add_argument("--Ns", type=str, default="20,40,80,160", help="Comma-separated N values, e.g. '20,40,80,160'")
    parser.add_argument("--T", type=float, default=1.0, help="Final time T")
    parser.add_argument("--out", type=str, default="hw1/error_plot_implicit.png", help="Output path for saved plot")
    args = parser.parse_args()

    Ns = [int(s) for s in args.Ns.split(",")]
    T = args.T

    results: List[RunResult] = []
    for N in Ns:
        res = solve_heat_backward_euler(N, T)
        results.append(res)
        print(f"N={res.N:4d}  dx={res.dx:.6e}  M={res.M:7d}  dt={res.dt:.6e}  err_max={res.err_max:.6e}")

    results.sort(key=lambda r: r.dx)
    dxs = np.array([r.dx for r in results])
    errs = np.array([r.err_max for r in results])

    plt.figure(figsize=(6,4))
    plt.loglog(dxs, errs, 'o-', label='Backward Euler error')

    # reference slope-2 line through last point
    c2 = errs[-1] / (dxs[-1] ** 2)
    plt.loglog(dxs, c2 * dxs**2, 'k--', label='O(dx^2)')

    plt.xlabel('dx = pi/N')
    plt.ylabel('max_j |u_j^M - u(x_j,T)|')
    plt.title('Implicit (Backward Euler) convergence: heat eq. on [0, pi]')
    plt.grid(True, which='both', ls=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved log-log error plot to: {args.out}")


if __name__ == "__main__":
    main()
