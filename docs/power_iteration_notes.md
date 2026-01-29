# Power Iteration Method: Technical Notes

## How the Double VJP Trick Works

The power iteration method finds the top-k right singular vectors of the Jacobian J = ∂target/∂steering_vec without ever forming J explicitly.

### The Jacobian

- **Input**: steering vector `sv` of shape `[hidden_dim]` (5120 for Qwen3-14B)
- **Output**: target activations at last k token positions, flattened to `[k * hidden_dim]`
- **J** has shape `[k * hidden_dim, hidden_dim]`

### The Algorithm

```python
sv = torch.zeros(hidden_dim, requires_grad=True)  # zeros - just scaffolding
u = torch.zeros_like(target, requires_grad=True)   # zeros - just scaffolding
v = V[:, col]  # current singular vector estimate (THIS is what we're computing)

model(inputs["input_ids"])
target = captured["target"]

t_slice = target[:, -4:, :].reshape(-1)  # last 4 tokens flattened
u_slice = u[:, -4:, :].reshape(-1)

# Step 1: Build computation graph for J^T @ u
grad = autograd.grad(t_slice, sv, grad_outputs=u_slice, create_graph=True)[0]

# Step 2: Differentiate w.r.t. u to get J, apply to v → gives J @ v
jvp = autograd.grad(grad, u_slice, grad_outputs=v)[0]

# Step 3: Apply J^T to get J^T @ (J @ v) = (J^T J) @ v
new_v = autograd.grad(t_slice, sv, grad_outputs=jvp.detach())[0]
```

### Why Zeros Work

**Q: Wouldn't J^T @ 0 = 0?**

A: Yes, but we're not evaluating the function at zeros - we're computing the *derivative* of the function. With `create_graph=True`, PyTorch builds a graph representing "grad as a function of u_slice", even when u_slice=0. The derivative of a linear function `f(u) = J^T @ u` is `df/du = J^T`, regardless of what u is.

**Q: What role does v play?**

A: `v` is the current estimate of the singular vector from the previous iteration. It's the actual state being refined by power iteration:
- Initialize v randomly
- Apply (J^T J) to v
- Normalize
- Repeat → converges to top right singular vector

**Q: Do sv, u, or v initial values matter?**

| Variable | Initial Value | Matters? | Why |
|----------|--------------|----------|-----|
| sv | zeros | No | Just the point where Jacobian is evaluated. For linear perturbation, J is constant. |
| u | zeros | No | Just scaffolding to build computation graph. Derivative doesn't depend on evaluation point. |
| v | random | **Yes** | This is the state being iterated. Converges to top singular vector. |

## Potential Extensions

### 1. Compute Jacobian at a Steered Point

Currently we compute J at sv=0 (unsteered). What if we set sv to a known steering vector (e.g., from CAA)?

```python
sv = caa_vector.clone().requires_grad_(True)  # instead of zeros
```

This would compute:
```
J_steered = ∂target/∂sv |_{sv=CAA_vec}
```

**Why this might be interesting:**
- Transformers have nonlinearities (LayerNorm, attention softmax, GeLU)
- Jacobian might differ at steered vs unsteered points
- Could find directions that complement an existing steering vector
- Could reveal nonlinear interactions between steering directions

### 2. Initialize v with CAA Vector

Instead of random initialization for v, start with a CAA vector:

```python
V = caa_vector.unsqueeze(1)  # use CAA as initial guess
V = orthogonalize(V)
```

This biases power iteration to find singular vectors aligned with CAA - useful if CAA is in the right neighborhood.

### 3. Questions to Investigate

1. **Does the Jacobian change significantly at steered vs unsteered points?**
   - Compare singular values/vectors at sv=0 vs sv=steering_vec
   - Large differences would indicate nonlinear steering effects

2. **Do singular vectors at a steered point complement the steering vector?**
   - Could lead to better multi-vector steering strategies

3. **Is there a relationship between MELBO vectors and top singular vectors?**
   - MELBO maximizes ||J @ v|| directly via gradient ascent
   - Power iteration finds directions that maximize ||J @ v|| / ||v|| (singular vectors)
   - Are they finding the same directions?

## Token Position Choice

- **MELBO**: last 2 tokens
- **Power iteration**: last 4 tokens

This is a hyperparameter choice. Last k tokens are most relevant for next-token prediction. Could experiment with:
- Different values of k
- Weighted combination of positions
- All positions (but more expensive)

## Relationship to MELBO

Both methods try to find steering directions with maximum effect on target activations:

| Method | Objective | Approach |
|--------|-----------|----------|
| MELBO | Maximize \|\|steered - unsteered\|\| | Gradient ascent on steering vector |
| Power Iter | Find top singular vectors of J | Iterative (J^T J) application |

MELBO directly optimizes for maximum displacement. Power iteration finds orthogonal directions ordered by their "sensitivity" (singular values).

In practice, MELBO vectors showed stronger steering effects, but power iteration found some complementary directions (e.g., vec 5 worked better than any single MELBO vector for survival at positive scales).
