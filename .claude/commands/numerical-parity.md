# Numerical Parity

Verify numerical equivalence between two implementations or between a reference and current implementation.

## Arguments

- `$ARGUMENTS` — Two file paths or a file path + reference description (e.g., "src/models/attention.py vs reference/attention_ref.py" or "src/models/mha.py check against PyTorch nn.MultiheadAttention")

## Instructions

1. Parse `$ARGUMENTS` to identify the two implementations to compare:
   - **Two files**: Compare file A vs file B
   - **File + reference**: Compare file A against a known reference implementation

2. Read both implementations. Identify:
   - Input/output signatures
   - Key numerical operations (matmul, softmax, normalization, activation functions)
   - Random seed handling
   - Dtype and device assumptions

3. Generate a **parity test script** that:

```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Tolerance settings
ATOL = 1e-5   # absolute tolerance
RTOL = 1e-4   # relative tolerance

def check_parity():
    """Compare outputs of both implementations."""
    # Create identical inputs
    x = torch.randn(<shape>, dtype=torch.float32)

    # Run implementation A
    model_a = <ImplA>(<params>)
    out_a = model_a(x)

    # Run implementation B
    model_b = <ImplB>(<params>)
    # Copy weights from A to B if applicable
    out_b = model_b(x)

    # Compare
    max_abs_diff = (out_a - out_b).abs().max().item()
    max_rel_diff = ((out_a - out_b) / (out_b.abs() + 1e-8)).abs().max().item()
    allclose = torch.allclose(out_a, out_b, atol=ATOL, rtol=RTOL)

    print(f"Max absolute diff: {max_abs_diff:.2e}")
    print(f"Max relative diff: {max_rel_diff:.2e}")
    print(f"All close (atol={ATOL}, rtol={RTOL}): {allclose}")

    # Check gradient parity
    loss_a = out_a.sum()
    loss_b = out_b.sum()
    loss_a.backward()
    loss_b.backward()

    for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if pa.grad is not None and pb.grad is not None:
            grad_close = torch.allclose(pa.grad, pb.grad, atol=ATOL, rtol=RTOL)
            if not grad_close:
                diff = (pa.grad - pb.grad).abs().max().item()
                print(f"Grad mismatch [{na}]: max diff = {diff:.2e}")

    return allclose

if __name__ == "__main__":
    check_parity()
```

4. Run the parity test script and report results.
5. If parity fails:
   - Identify which operations diverge
   - Check for common issues: different initialization, broadcasting bugs, transpose errors, dtype mismatch
   - Suggest specific fixes
6. Also check:
   - **FP16/BF16 parity**: If applicable, test with lower precision
   - **Batch dimension**: Test with batch_size=1 and batch_size>1
   - **Edge cases**: Zero inputs, very large values, sequence length=1

## Output

Print a structured parity report:
```
=== Numerical Parity Report ===
Implementation A: <path>
Implementation B: <path>
Forward pass:  PASS/FAIL (max diff: X.XXe-XX)
Backward pass: PASS/FAIL (max grad diff: X.XXe-XX)
FP16 forward:  PASS/FAIL (if tested)
Edge cases:    PASS/FAIL
```
