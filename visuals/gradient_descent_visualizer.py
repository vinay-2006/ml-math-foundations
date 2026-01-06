"""
Gradient Descent Visualizer (Engineering-Grade)
-----------------------------------------------

Purpose:
- Visualize Gradient Descent as geometric motion on a loss surface
- Debug convergence vs divergence
- Reuse the same optimizer for different loss functions
- Reinforce correct stopping conditions based on gradient magnitude

This tool is Week-B ready and portfolio-safe.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DEFAULT LOSS FUNCTION AND GRADIENT (1D FOR VISUALIZATION)
# ============================================================

def default_loss(w):
    """
    Simple convex loss function.
    Global minimum at w = 3.
    """
    return (w - 3) ** 2


def default_gradient(w):
    """
    Analytical gradient of the default loss.
    """
    return 2 * (w - 3)


# ============================================================
# 2. GENERIC GRADIENT DESCENT ENGINE
# ============================================================

def gradient_descent(
    start_w: float,
    learning_rate: float,
    max_steps: int,
    loss_fn,
    grad_fn,
    tol: float = 1e-6
):
    """
    Generic Gradient Descent Engine.

    Parameters
    ----------
    start_w : float
        Initial parameter value.
    learning_rate : float
        Step size (η).
    max_steps : int
        Maximum number of iterations.
    loss_fn : callable
        Loss function L(w).
    grad_fn : callable
        Gradient function dL/dw.
    tol : float
        Gradient magnitude threshold for convergence.

    Returns
    -------
    history_w : list
        Parameter values over iterations.
    history_loss : list
        Loss values over iterations.
    """

    w = start_w
    history_w = []
    history_loss = []

    for step in range(max_steps):
        grad = grad_fn(w)

        history_w.append(w)
        history_loss.append(loss_fn(w))

        # 🔒 Engineering stop condition (convergence)
        if abs(grad) < tol:
            print(f"Converged in {step} steps (|grad| < {tol})")
            break

        # Core GD update rule (Week-B power line)
        w = w - learning_rate * grad

    return history_w, history_loss


# ============================================================
# 3. VISUALIZATION FUNCTION
# ============================================================

def visualize_gradient_descent(
    start_w=0.0,
    learning_rate=0.1,
    max_steps=20,
    loss_fn=default_loss,
    grad_fn=default_gradient
):
    """
    Visualize gradient descent as motion on a loss curve.
    """

    # Loss surface
    w_space = np.linspace(-1, 7, 400)
    loss_space = [loss_fn(w) for w in w_space]

    # Run gradient descent
    history_w, history_loss = gradient_descent(
        start_w=start_w,
        learning_rate=learning_rate,
        max_steps=max_steps,
        loss_fn=loss_fn,
        grad_fn=grad_fn
    )

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(w_space, loss_space, label="Loss Curve", linewidth=2)

    # Plot GD trajectory
    for i in range(len(history_w)):
        plt.scatter(history_w[i], history_loss[i], color="red")
        if i > 0:
            plt.plot(
                [history_w[i - 1], history_w[i]],
                [history_loss[i - 1], history_loss[i]],
                color="red",
                linewidth=1
            )

    plt.xlabel("Parameter (w)")
    plt.ylabel("Loss")
    plt.title(
        f"Gradient Descent Visualization\n"
        f"start_w={start_w}, lr={learning_rate}"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 4. LEARNING RATE COMPARISON UTILITY
# ============================================================

def compare_learning_rates(
    learning_rates=(0.01, 0.1, 0.5, 1.1),
    max_steps=25,
    loss_fn=default_loss,
    grad_fn=default_gradient
):
    """
    Compare convergence behavior for different learning rates.
    """

    plt.figure(figsize=(8, 5))

    for lr in learning_rates:
        _, losses = gradient_descent(
            start_w=0.0,
            learning_rate=lr,
            max_steps=max_steps,
            loss_fn=loss_fn,
            grad_fn=grad_fn
        )
        plt.plot(losses, label=f"lr={lr}")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Effect of Learning Rate on Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 5. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Single run visualization
    visualize_gradient_descent(
        start_w=0.0,
        learning_rate=0.1,
        max_steps=20
    )

    # Learning rate comparison
    compare_learning_rates()
