import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

"""
**5. Coding problem:**

Consider matrix $ A $ from Problem 4, and let $ N = 49, 99, 199, 399 $. Write a code to implement the Gauss-Seidel method to solve the linear system $ Ax = b $, where the exact solution is $ x \in \mathbb{R}^N $ with components $ x_j = \sin\left(\frac{10j\pi}{N}\right) $, and $ b = Ax $.

Set the maximum number of iterations to be 50,000, and use the zero vector as the initial guess. The stopping criterion is:
$$
\frac{\|r^{(k)}\|_2}{\|b\|_2} < 1 \times 10^{-8},
$$
where the residual at iteration $ k $ is defined as $ r^{(k)} = Ax^{(k)} - b $.

Plot the history of the residual norm $ \|r^{(k)}\|_2 $ (on a log scale if appropriate) for each value of $ N $. Also, output the final error $ \|x^{(K)} - x\| $, where $ K $ is the number of iterations when the Gauss-Seidel iteration stops.

"""

def create_matrix_A(N):
    """
    Create matrix A from Problem 4.
    Assuming it's a tridiagonal matrix: A = tridiag(-1, 2, -1) / h^2
    where h = 1/(N+1)
    """
    h = 1.0 / (N + 1)
    diagonals = [-1.0/h**2, 2.0/h**2, -1.0/h**2]
    offsets = [-1, 0, 1]
    A = diags(diagonals, offsets, shape=(N, N)).toarray()
    return A

def gauss_seidel_optimized(A, b, x0, max_iter=50000, tol=1e-8):
    """
    Optimized Gauss-Seidel method to solve Ax = b
    Uses vectorized operations for tridiagonal matrices
    """
    N = len(b)
    x = x0.copy()
    residual_history = []
    
    b_norm = np.linalg.norm(b)
    
    # Extract diagonals for faster computation
    main_diag = np.diag(A)
    lower_diag = np.diag(A, -1)
    upper_diag = np.diag(A, 1)
    
    for k in range(max_iter):
        # Compute residual every 10 iterations to save time
        if k % 10 == 0:
            r = A @ x - b
            r_norm = np.linalg.norm(r)
            residual_history.append(r_norm)
            
            # Check stopping criterion
            if r_norm / b_norm < tol:
                # Compute final residual
                r = A @ x - b
                r_norm = np.linalg.norm(r)
                residual_history.append(r_norm)
                print(f"Converged in {k} iterations")
                return x, residual_history, k
        
        # Gauss-Seidel iteration optimized for tridiagonal matrix
        # First element
        x[0] = (b[0] - upper_diag[0] * x[1]) / main_diag[0]
        
        # Middle elements
        for i in range(1, N-1):
            x[i] = (b[i] - lower_diag[i-1] * x[i-1] - upper_diag[i] * x[i+1]) / main_diag[i]
        
        # Last element
        x[N-1] = (b[N-1] - lower_diag[N-2] * x[N-2]) / main_diag[N-1]
    
    # Final residual
    r = A @ x - b
    r_norm = np.linalg.norm(r)
    residual_history.append(r_norm)
    print(f"Reached maximum iterations ({max_iter})")
    return x, residual_history, max_iter

def main():
    # Values of N to test
    N_values = [49, 99, 199, 399]
    
    # Create figure for plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    print("Gauss-Seidel Method Results")
    print("=" * 60)
    
    results = []
    
    for idx, N in enumerate(N_values):
        print(f"\nN = {N}")
        print("-" * 60)
        
        # Create matrix A
        A = create_matrix_A(N)
        
        # Create exact solution x
        x_exact = np.array([np.sin(10 * j * np.pi / N) for j in range(1, N + 1)])
        
        # Compute b = Ax
        b = A @ x_exact
        
        # Initial guess (zero vector)
        x0 = np.zeros(N)
        
        # Solve using Gauss-Seidel
        x_computed, residual_history, num_iter = gauss_seidel_optimized(A, b, x0)
        
        # Compute final error
        error = np.linalg.norm(x_computed - x_exact)
        
        # Output results
        print(f"Number of iterations: {num_iter}")
        print(f"Final error ||x^(K) - x||_2: {error:.6e}")
        print(f"Final residual ||r^(K)||_2: {residual_history[-1]:.6e}")
        
        results.append({
            'N': N,
            'iterations': num_iter,
            'error': error,
            'residual': residual_history[-1]
        })
        
        # Plot residual history
        iterations = np.arange(0, len(residual_history) * 10, 10)[:len(residual_history)]
        axes[idx].semilogy(iterations, residual_history, 'b-', linewidth=1.5)
        axes[idx].set_xlabel('Iteration', fontsize=10)
        axes[idx].set_ylabel('Residual Norm ||r^(k)||_2', fontsize=10)
        axes[idx].set_title(f'N = {N}', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/automatocti/Documents/ust/course/Math5311/MATH5311/hw4/gauss_seidel_convergence.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Plot saved as 'gauss_seidel_convergence.png'")
    
    # Save summary table to txt file
    summary_path = '/Users/automatocti/Documents/ust/course/Math5311/MATH5311/hw4/gauss_seidel_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"{'N':<10} {'Iterations':<15} {'Final Error':<25} {'Final Residual':<25}\n")
        f.write("-" * 75 + "\n")
        for res in results:
            f.write(f"{res['N']:<10} {res['iterations']:<15} {res['error']:<25.6e} {res['residual']:<25.6e}\n")
            
    print(f"Summary table saved as '{summary_path}'")
    plt.show()

if __name__ == "__main__":
    main()
