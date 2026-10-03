from collections.abc import Sequence

import torch
from torch import Tensor, nn


class RegressionHead(nn.Module):
    """Predicts one continuous target from a subject embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initializes a task-specific regression head.

        Args:
            input_dim: Dimension of the subject embedding.
            hidden_dim: Hidden dimension of the task-specific multilayer
                perceptron.
            dropout: Dropout applied before the final prediction layer.
                Defaults to 0.1.

        Raises:
            ValueError: If a dimension or dropout value is invalid.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, subject_embedding: Tensor) -> Tensor:
        """Predicts one task value.

        Args:
            subject_embedding: Participant representations with shape
                ``[batch_size, input_dim]``.

        Returns:
            Predicted values with shape ``[batch_size]``.

        Raises:
            ValueError: If the subject embedding is not two-dimensional.
        """
        if subject_embedding.ndim != 2:
            raise ValueError("subject_embedding must have shape [batch_size, input_dim].")

        return self.network(subject_embedding).squeeze(-1)


class MultiTaskRegressionHead(nn.Module):
    """Uses an independent regression head for each prediction task.

    A ``ModuleList`` is intentionally used instead of a ``ModuleDict`` because
    UK Biobank task names can contain periods, such as
    ``participant.p20016_i2``. PyTorch module names cannot contain periods.
    """

    def __init__(
        self,
        input_dim: int,
        task_names: Sequence[str],
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the multi-task regression heads.

        Args:
            input_dim: Dimension of the participant representation.
            task_names: Ordered task names. The resulting prediction columns
                follow this exact order.
            hidden_dim: Hidden dimension of every task-specific head. Defaults
                to half of ``input_dim``, with a minimum value of one.
            dropout: Dropout used inside each task-specific head. Defaults to
                0.1.

        Raises:
            ValueError: If no tasks are provided, task names are duplicated, or
                dimensions are invalid.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        normalized_task_names = tuple(str(task_name) for task_name in task_names)

        if not normalized_task_names:
            raise ValueError("task_names must contain at least one task.")

        if any(not task_name for task_name in normalized_task_names):
            raise ValueError("task_names cannot contain empty names.")

        if len(set(normalized_task_names)) != len(normalized_task_names):
            raise ValueError("task_names must be unique.")

        if hidden_dim is None:
            hidden_dim = max(1, input_dim // 2)

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.input_dim = input_dim
        self.task_names = normalized_task_names

        self.task_heads = nn.ModuleList(
            [
                RegressionHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in normalized_task_names
            ]
        )

    @property
    def num_tasks(self) -> int:
        """Returns the number of prediction tasks."""
        return len(self.task_names)

    def forward(self, subject_embedding: Tensor) -> Tensor:
        """Predicts all configured continuous targets.

        Args:
            subject_embedding: Participant-level representation with shape
                ``[batch_size, input_dim]``.

        Returns:
            Prediction tensor with shape ``[batch_size, num_tasks]``. Columns
            follow the order stored in ``task_names``.

        Raises:
            ValueError: If the input tensor shape is invalid.
        """
        if subject_embedding.ndim != 2:
            raise ValueError("subject_embedding must have shape [batch_size, input_dim].")

        if subject_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected subject embedding dimension {self.input_dim}, but received {subject_embedding.shape[-1]}."
            )

        predictions = [task_head(subject_embedding) for task_head in self.task_heads]

        return torch.stack(predictions, dim=-1)

    def predictions_to_dict(
        self,
        predictions: Tensor,
    ) -> dict[str, Tensor]:
        """Maps prediction columns to their task names.

        Args:
            predictions: Prediction tensor with shape
                ``[batch_size, num_tasks]``.

        Returns:
            Mapping from each task name to its prediction vector.

        Raises:
            ValueError: If the prediction shape does not match the configured
                number of tasks.
        """
        if predictions.ndim != 2:
            raise ValueError("predictions must have shape [batch_size, num_tasks].")

        if predictions.shape[1] != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} prediction tasks, but received {predictions.shape[1]}.")

        return {task_name: predictions[:, task_index] for task_index, task_name in enumerate(self.task_names)}


class TaskGatedRegressionHead(nn.Module):
    """Uses interpretable task-specific mixtures of shared and regional experts.

    The first expert token is the global/shared representation. Remaining tokens are regional expert representations.
    Each task learns one static softmax gate over these tokens and one linear regression head.
    """

    def __init__(
        self,
        input_dim: int,
        num_expert_tokens: int,
        task_names: Sequence[str],
    ) -> None:
        """Initializes task gates and minimal linear prediction heads."""
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_expert_tokens <= 1:
            raise ValueError("num_expert_tokens must include one shared and at least one regional expert.")
        normalized_task_names = tuple(str(task_name) for task_name in task_names)
        if not normalized_task_names or len(set(normalized_task_names)) != len(normalized_task_names):
            raise ValueError("task_names must be non-empty and unique.")

        self.input_dim = input_dim
        self.num_expert_tokens = num_expert_tokens
        self.task_names = normalized_task_names
        self.task_gate_logits = nn.Parameter(torch.zeros(len(normalized_task_names), num_expert_tokens))
        self.task_norms = nn.ModuleList(nn.LayerNorm(input_dim) for _ in normalized_task_names)
        self.task_heads = nn.ModuleList(nn.Linear(input_dim, 1) for _ in normalized_task_names)

    @property
    def num_tasks(self) -> int:
        """Returns the number of prediction tasks."""
        return len(self.task_names)

    def normalized_task_gates(self) -> Tensor:
        """Returns learned task-to-expert weights with shape [tasks, experts]."""
        return torch.softmax(self.task_gate_logits, dim=-1)

    def forward(
        self,
        expert_tokens: Tensor,
        expert_token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns predictions, task representations, and effective gates."""
        if expert_tokens.ndim != 3:
            raise ValueError("expert_tokens must have shape [batch_size, num_expert_tokens, input_dim].")
        batch_size, num_expert_tokens, input_dim = expert_tokens.shape
        if (num_expert_tokens, input_dim) != (self.num_expert_tokens, self.input_dim):
            raise ValueError(
                "expert token shape does not match the configured head: "
                f"expected (*, {self.num_expert_tokens}, {self.input_dim}), received {tuple(expert_tokens.shape)}."
            )

        gates = self.normalized_task_gates().unsqueeze(0).expand(batch_size, -1, -1)
        if expert_token_mask is not None:
            if tuple(expert_token_mask.shape) != (batch_size, num_expert_tokens):
                raise ValueError("expert_token_mask must have shape [batch_size, num_expert_tokens].")
            gates = gates * expert_token_mask.to(device=expert_tokens.device, dtype=expert_tokens.dtype).unsqueeze(1)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        task_representations = torch.einsum("bte,bed->btd", gates, expert_tokens)
        predictions = torch.cat(
            [
                task_head(task_norm(task_representations[:, task_index])).reshape(batch_size, 1)
                for task_index, (task_norm, task_head) in enumerate(zip(self.task_norms, self.task_heads, strict=True))
            ],
            dim=1,
        )
        return predictions, task_representations, gates


class AdditiveExpertRegressionHead(nn.Module):
    """Predicts age as an exact sum of independently projected expert tokens.

    Expert tokens are normalized and projected before they are combined. The returned scalar contributions therefore
    reconstruct every prediction exactly, unlike a head that normalizes an already mixed representation.
    """

    def __init__(
        self,
        input_dim: int,
        num_expert_tokens: int,
        task_names: Sequence[str],
    ) -> None:
        """Initialize task-specific gates, projections, and intercepts."""
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_expert_tokens <= 1:
            raise ValueError("num_expert_tokens must include one shared and at least one regional expert.")
        normalized_task_names = tuple(str(task_name) for task_name in task_names)
        if not normalized_task_names or len(set(normalized_task_names)) != len(normalized_task_names):
            raise ValueError("task_names must be non-empty and unique.")

        self.input_dim = input_dim
        self.num_expert_tokens = num_expert_tokens
        self.task_names = normalized_task_names
        self.task_gate_logits = nn.Parameter(torch.zeros(len(normalized_task_names), num_expert_tokens))
        self.task_norms = nn.ModuleList(nn.LayerNorm(input_dim) for _ in normalized_task_names)
        self.task_projections = nn.ModuleList(nn.Linear(input_dim, 1, bias=False) for _ in normalized_task_names)
        self.task_biases = nn.Parameter(torch.zeros(len(normalized_task_names)))

    @property
    def num_tasks(self) -> int:
        """Return the number of prediction tasks."""
        return len(self.task_names)

    def normalized_task_gates(self) -> Tensor:
        """Return learned task-to-expert weights with shape [tasks, experts]."""
        return torch.softmax(self.task_gate_logits, dim=-1)

    def forward(
        self,
        expert_tokens: Tensor,
        expert_token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return predictions, representations, gates, contributions, and biases."""
        if expert_tokens.ndim != 3:
            raise ValueError("expert_tokens must have shape [batch_size, num_expert_tokens, input_dim].")
        batch_size, num_expert_tokens, input_dim = expert_tokens.shape
        if (num_expert_tokens, input_dim) != (self.num_expert_tokens, self.input_dim):
            raise ValueError(
                "expert token shape does not match the configured head: "
                f"expected (*, {self.num_expert_tokens}, {self.input_dim}), received {tuple(expert_tokens.shape)}."
            )

        gates = self.normalized_task_gates().unsqueeze(0).expand(batch_size, -1, -1)
        if expert_token_mask is not None:
            if tuple(expert_token_mask.shape) != (batch_size, num_expert_tokens):
                raise ValueError("expert_token_mask must have shape [batch_size, num_expert_tokens].")
            gates = gates * expert_token_mask.to(device=expert_tokens.device, dtype=expert_tokens.dtype).unsqueeze(1)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        task_representations = torch.einsum("bte,bed->btd", gates, expert_tokens)
        contribution_columns = []
        for task_norm, task_projection in zip(self.task_norms, self.task_projections, strict=True):
            expert_scores = task_projection(task_norm(expert_tokens)).squeeze(-1)
            task_index = len(contribution_columns)
            contribution_columns.append(gates[:, task_index, :] * expert_scores)
        contributions = torch.stack(contribution_columns, dim=1)
        intercepts = self.task_biases.unsqueeze(0).expand(batch_size, -1)
        predictions = intercepts + contributions.sum(dim=-1)

        if not torch.allclose(predictions, intercepts + contributions.sum(dim=-1), atol=1e-6, rtol=1e-6):
            raise RuntimeError("Additive expert contributions failed to reconstruct predictions.")
        return predictions, task_representations, gates, contributions, intercepts
