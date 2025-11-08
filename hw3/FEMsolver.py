r"""Finite element solver for -u'' = 2 on (0, 1) with a staggered mesh.

The mesh is defined by N interior nodes with spacing \Delta x = 1/(N+1).
Interior node locations obey

	x_j = j*\Delta x                when j is even,
	x_j = (j - 1/2)*\Delta x        when j is odd,

for j = 1, ..., N, while x_0 = 0 and x_{N+1} = 1.  Dirichlet data is zero at
both endpoints.  The exact solution is u(x) = x(1 - x).

The code below assembles the FEM system with piecewise linear basis functions,
solves for the nodal values, and reports the maximum nodal error for several
choices of N.  It also produces a log-log convergence plot of the error versus
the representative mesh size h = max_j (x_j - x_{j-1}).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FemRunResult:
	"""Container for a single FEM run."""

	N: int
	nodes: np.ndarray
	interior_solution: np.ndarray
	max_error: float


def build_mesh(N: int) -> np.ndarray:
	"""Return the nodal coordinates for the staggered mesh.

	The returned array has length N + 2 and includes the boundary points 0 and 1.
	"""

	if N < 1:
		raise ValueError("N must be positive")

	dx = 1.0 / (N + 1)
	nodes = np.zeros(N + 2, dtype=float)
	for j in range(1, N + 2):
		if j % 2 == 0:
			nodes[j] = j * dx
		else:
			nodes[j] = (j - 0.5) * dx
	nodes[-1] = 1.0  # ensure the right boundary lands exactly at 1
	return nodes


def assemble_system(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Assemble the stiffness matrix and load vector for interior nodes."""

	N = nodes.size - 2
	K = np.zeros((N, N), dtype=float)
	f = np.zeros(N, dtype=float)

	# Element lengths h_j = x_j - x_{j-1} for j = 1, ..., N+1
	h = np.diff(nodes)

	for i in range(1, N + 1):
		row = i - 1
		h_left = h[i - 1]
		h_right = h[i]

		K[row, row] = 1.0 / h_left + 1.0 / h_right
		if row > 0:
			K[row, row - 1] = -1.0 / h_left
		if row < N - 1:
			K[row, row + 1] = -1.0 / h_right

		# Load vector contribution for constant source term f(x) = 2
		f[row] = h_left + h_right

	return K, f


def solve_fem(N: int) -> FemRunResult:
	"""Solve the FEM system for a given number of interior nodes N."""

	nodes = build_mesh(N)
	K, rhs = assemble_system(nodes)
	interior_solution = np.linalg.solve(K, rhs)

	full_solution = np.zeros_like(nodes)
	full_solution[1:-1] = interior_solution

	exact = nodes[1:-1] * (1.0 - nodes[1:-1])
	error = np.abs(full_solution[1:-1] - exact)
	max_err = float(np.max(error))

	return FemRunResult(N=N, nodes=nodes, interior_solution=interior_solution, max_error=max_err)


def run_study(N_values: Iterable[int]) -> list[FemRunResult]:
	"""Solve the FEM problem for each N in N_values and collect results."""

	results: list[FemRunResult] = []
	for N in N_values:
		result = solve_fem(N)
		results.append(result)
	return results


def plot_errors(results: list[FemRunResult], save_path: Optional[Path] = None) -> None:
	"""Plot max nodal error versus representative mesh size."""

	h_values = []
	errors = []
	for res in results:
		# Representative h is the maximum element size on the mesh.
		h_local = np.max(np.diff(res.nodes))
		h_values.append(h_local)
		errors.append(res.max_error)

	fig = plt.figure(figsize=(6, 4))
	ax = fig.add_subplot(1, 1, 1)
	ax.loglog(h_values, errors, marker="o", label="FEM error")

	# Reference line with slope 2 (expected second-order accuracy on this mesh)
	slope = 2
	ref_h = np.array(h_values)
	ref_line = errors[0] * (ref_h / ref_h[0]) ** slope
	ax.loglog(ref_h, ref_line, linestyle="--", label=f"O(h^{slope})")

	ax.invert_xaxis()
	ax.set_xlabel("max element size h")
	ax.set_ylabel("max nodal error")
	ax.set_title("FEM convergence for -u'' = 2")
	ax.legend()
	ax.grid(True, which="both", linewidth=0.5, alpha=0.6)
	fig.tight_layout()

	if save_path is not None:
		fig.savefig(save_path, dpi=300)

	plt.show()


def main() -> None:
	N_values = [39, 79, 159, 319]
	results = run_study(N_values)

	header = f"{'N':>6}  {'max error':>12}  {'max h':>12}"
	print(header)
	print("-" * len(header))
	for res in results:
		max_h = float(np.max(np.diff(res.nodes)))
		print(f"{res.N:6d}  {res.max_error:12.6e}  {max_h:12.6e}")

	figure_path = Path(__file__).with_name("fem_error_plot.png")
	plot_errors(results, save_path=figure_path)
	print(f"Saved error plot to {figure_path}")


if __name__ == "__main__":
	main()
