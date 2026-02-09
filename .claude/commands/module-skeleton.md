# Module Skeleton

Generate a new Python module skeleton for an AI/ML model component.

## Arguments

- `$ARGUMENTS` — Module name and brief description (e.g., "attention multi-head attention layer")

## Instructions

1. Parse the module name and description from `$ARGUMENTS`.
2. Determine the appropriate location under the project `src/` or `models/` directory. If neither exists, create `src/models/`.
3. Generate the module file with the following structure:

```python
"""
<Module docstring: one-line description>
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

__all__ = ["<ClassName>"]


class <ClassName>(nn.Module):
    """<Brief description>.

    Args:
        <list constructor params with types and defaults>
    """

    def __init__(self, <params>):
        super().__init__()
        # TODO: initialize layers

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        return "<param summary>"
```

4. Also generate a corresponding `__init__.py` update to export the new class.
5. If a test directory exists (`tests/`), generate a minimal test stub file `test_<module>.py`:

```python
import pytest
import torch
from <package>.<module> import <ClassName>


class Test<ClassName>:
    def test_forward_shape(self):
        """Verify output shape matches expected dimensions."""
        pass  # TODO: implement

    def test_gradient_flow(self):
        """Verify gradients flow through the module."""
        pass  # TODO: implement
```

6. Print a summary of created files.
