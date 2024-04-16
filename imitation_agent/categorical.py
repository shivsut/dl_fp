import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits

import warnings
from typing import Optional, Tuple
from torch.types import _size


class Distribution:

    _validate_args = __debug__

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size(),
        event_shape: torch.Size = torch.Size(),
        validate_args: Optional[bool] = None,
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        if validate_args is not None:
            self._validate_args = validate_args
        if self._validate_args:
            try:
                arg_constraints = self.arg_constraints
            except NotImplementedError:
                arg_constraints = {}
                warnings.warn(
                    f"{self.__class__} does not define `arg_constraints`. "
                    + "Please set `arg_constraints = {}` or initialize the distribution "
                    + "with `validate_args=False` to turn off validation."
                )
            for param, constraint in arg_constraints.items():
                if constraints.is_dependent(constraint):
                    continue  # skip constraints that cannot be checked
                if param not in self.__dict__ and isinstance(
                    getattr(type(self), param), lazy_property
                ):
                    continue  # skip checking lazily-constructed args
                value = getattr(self, param)
                valid = constraint.check(value)
                if not valid.all():
                    raise ValueError(
                        f"Expected parameter {param} "
                        f"({type(value).__name__} of shape {tuple(value.shape)}) "
                        f"of distribution {repr(self)} "
                        f"to satisfy the constraint {repr(constraint)}, "
                        f"but found invalid values:\n{value}"
                    )

    def _extended_shape(self, sample_shape: _size = torch.Size()) -> Tuple[int, ...]:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return torch.Size(sample_shape + self._batch_shape + self._event_shape)

    def _validate_sample(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise ValueError("The value argument to log_prob must be a Tensor")

        event_dim_start = len(value.size()) - len(self._event_shape)
        if value.size()[event_dim_start:] != self._event_shape:
            raise ValueError(
                f"The right-most size of value must match event_shape: {value.size()} vs {self._event_shape}."
            )

        actual_shape = value.size()
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError(
                    f"Value is not broadcastable with batch_shape+event_shape: {actual_shape} vs {expected_shape}."
                )
        try:
            support = self.support
        except NotImplementedError:
            warnings.warn(
                f"{self.__class__} does not define `support` to enable "
                + "sample validation. Please initialize the distribution with "
                + "`validate_args=False` to turn off validation."
            )
            return
        assert support is not None
        valid = support.check(value)
        if not valid.all():
            raise ValueError(
                "Expected value argument "
                f"({type(value).__name__} of shape {tuple(value.shape)}) "
                f"to be within the support ({repr(support)}) "
                f"of the distribution {repr(self)}, "
                f"but found invalid values:\n{value}"
            )

    def _get_checked_instance(self, cls, _instance=None):
        if _instance is None and type(self).__init__ != cls.__init__:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} of {cls.__name__} that defines a custom __init__ method "
                "must also define a custom .expand() method."
            )
        return self.__new__(type(self)) if _instance is None else _instance


class Categorical(Distribution):
   
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    # has_enumerate_support = True

    def __init__(self, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = (
            self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        )
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Categorical, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        new._num_events = self._num_events
        super(Categorical, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)

    def logits(self):
        with torch.enable_grad():
            output = probs_to_logits(self.probs)
        return output

    @lazy_property
    def probs(self):
        output = logits_to_probs(self.logits)
        return output

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)



