# ML Math Foundations — Day 01: Vector Mechanics

This repository contains the first step in a bottom-up reconstruction of machine learning: implementing and justifying the core vector operations used in linear prediction.

## 🎯 Implementation Goals
The focus of Day 01 is the manual implementation of vector operations to visualize how feature signals are combined before they reach an optimizer.

### 1. Manual Vector Operations (Pure Python)
Before utilizing optimized libraries, this module implements:
* **Vector Representation:** Mapping data points as lists where each index represents a specific feature.
* **Scalar Multiplication:** Demonstrating how changing magnitude impacts the "weight" of a feature signal.
* **Vector Addition:** Building the intuition for combining multiple feature signals into a single representation.

### 2. The Dot Product as a Score
The dot product is implemented manually to show exactly how a model generates a prediction (pre-bias).
* **Alignment:** Multiplying weights by features and summing them provides a single scalar "score".
* **Mechanic:** High scores indicate strong alignment between the input vector and the learned weight vector.

## 🛠 Technical Artifacts

| Operation | Manual Implementation (Intuition) | NumPy Verification (Scale) |
| :--- | :--- | :--- |
| **Vector Addition** | `[x1[0] + x2[0], x1[1] + x2[1]]` | `np.add(x1, x2)` |
| **Scalar Scaling** | `[scalar * x1[0], scalar * x1[1]]` | `scalar * x_np` |
| **Dot Product** | `w[0]*x[0] + w[1]*x[1]` | `np.dot(w, x)` |

## 🧪 Observation: The Scale Sensitivity
The manual implementation in `day01_vectors_dot_product.ipynb` reveals a critical property of the dot product: **it is sensitive to magnitude**.

* **Feature Dominance:** If one feature has a significantly larger scale, it will mathematically dominate the dot product, regardless of its importance.
* **Gradient Implication:** This observation justifies why feature scaling is a prerequisite for training; large magnitudes lead to large gradients, which can destabilize updates.

---
*Verified via Python 3.11 and NumPy.*