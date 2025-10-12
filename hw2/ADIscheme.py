import math
import sys
from dataclasses import dataclass
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ADIResult:
    x: np.ndarray
    y: np.ndarray
    solution: np.ndarray
    time: float


def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute u(x, y, 0) = sin(x) sin(y) on the grid."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.sin(X) * np.sin(Y)


def exact_solution(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Exact solution u(x, y, t) = exp(-2 t) sin(x) sin(y)."""
    return math.exp(-2.0 * t) * initial_condition(x, y)


def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system with the Thomas algorithm."""
    n = b.size
    c_prime = np.empty(n - 1, dtype=np.float64)
    d_prime = np.empty(n, dtype=np.float64)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    d_prime[-1] = (d[-1] - a[-1] * d_prime[-2]) / (b[-1] - a[-1] * c_prime[-2])

    x = np.empty(n, dtype=np.float64)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def adi_heat_solver(
    N: int,
    T: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ADIResult:
    """Advance the heat equation to time T using the ADI scheme on an N x N grid."""
    if N < 2:
        raise ValueError("N must be at least 2 to have interior grid points.")

    length = math.pi
    dx = length / N
    dy = length / N

    x = np.linspace(0.0, length, N + 1, dtype=np.float64)
    y = np.linspace(0.0, length, N + 1, dtype=np.float64)

    dt = dx
    n_steps = max(1, int(math.ceil(T / dt)))
    dt = T / n_steps

    rx = dt / (2.0 * dx * dx)
    ry = dt / (2.0 * dy * dy)

    u = initial_condition(x, y)
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    u_star = np.zeros_like(u)

    m = N - 1
    a_x = -rx * np.ones(m - 1, dtype=np.float64)
    b_x = (1.0 + 2.0 * rx) * np.ones(m, dtype=np.float64)
    c_x = -rx * np.ones(m - 1, dtype=np.float64)

    a_y = -ry * np.ones(m - 1, dtype=np.float64)
    b_y = (1.0 + 2.0 * ry) * np.ones(m, dtype=np.float64)
    c_y = -ry * np.ones(m - 1, dtype=np.float64)

    for step in range(n_steps):
        # First half-step: implicit in x, explicit in y
        for j in range(1, N):
            rhs = u[1:-1, j] + ry * (u[1:-1, j + 1] - 2.0 * u[1:-1, j] + u[1:-1, j - 1])
            u_star[1:-1, j] = thomas_algorithm(a_x, b_x, c_x, rhs)

        u_star[0, :] = 0.0
        u_star[-1, :] = 0.0
        u_star[:, 0] = 0.0
        u_star[:, -1] = 0.0

        # Second half-step: implicit in y, explicit in x
        for i in range(1, N):
            rhs = u_star[i, 1:-1] + rx * (u_star[i + 1, 1:-1] - 2.0 * u_star[i, 1:-1] + u_star[i - 1, 1:-1])
            u[i, 1:-1] = thomas_algorithm(a_y, b_y, c_y, rhs)

        u[0, :] = 0.0
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0

        if progress_callback is not None:
            progress_callback(step + 1, n_steps)

    return ADIResult(x=x, y=y, solution=u, time=T)


def compute_error(N: int, T: float, progress_callback: Callable[[int, int], None] | None = None) -> Tuple[float, float]:
    result = adi_heat_solver(N, T, progress_callback=progress_callback)
    x, y, approx = result.x, result.y, result.solution
    exact = exact_solution(x, y, result.time)
    interior_error = np.abs(approx[1:-1, 1:-1] - exact[1:-1, 1:-1])
    return math.pi / N, interior_error.max()


def make_progress_printer(N: int) -> Callable[[int, int], None]:
    bar_width = 20

    def _printer(step: int, total: int) -> None:
        filled = int(bar_width * step / total)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = 100.0 * step / total
        sys.stdout.write(f"\rN={N:3d} [{bar}] {percent:5.1f}%")
        if step == total:
            sys.stdout.write("\n")
        sys.stdout.flush()

    return _printer


def main() -> None:
    T = 1.0
    grid_sizes = [20, 40, 80, 160]

    hs = []
    errors = []

    for N in grid_sizes:
        progress = make_progress_printer(N)
        h, err = compute_error(N, T, progress_callback=progress)
        hs.append(h)
        errors.append(err)
        print(f"N={N:3d}, h={h:.5e}, max error={err:.5e}")

    hs = np.array(hs)
    errors = np.array(errors)

    coeffs = np.polyfit(np.log(hs), np.log(errors), 1)
    estimated_order = coeffs[0]

    plt.figure(figsize=(6, 4))
    plt.loglog(hs, errors, "o-", label="ADI max error")
    for h, err, N in zip(hs, errors, grid_sizes):
        plt.annotate(
            f"N={N}",
            xy=(h, err),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=9,
        )
    plt.loglog(hs, np.exp(coeffs[1]) * hs ** 2, "--", label="O(h^2)")
    plt.gca().invert_xaxis()
    plt.xlabel(r"$h$")
    plt.ylabel(r"$L_\infty$ error")
    plt.title("ADI scheme convergence at T=1")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig("adi_error_plot.png", dpi=200)
    plt.close()

    print(f"Estimated order of convergence: {estimated_order:.2f}")


if __name__ == "__main__":
    main()
