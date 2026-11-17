import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                             x: List[float],
                             W1: List[List[float]], b1: List[float],
                             W2: List[List[float]], b2: List[float],
                             y_true: List[float]) -> dict:
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

      
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        y_hat = W2 @ a1 + b2
        loss = np.mean((y_hat - y_true) ** 2)

    
        n = len(y_true)
        dz2 = 2 * (y_hat - y_true) / n
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        da1 = W2.T @ dz2
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }