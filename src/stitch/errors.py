"""Failure types whose recovery boundary is the rollout replica process."""


class UnrecoverableSidecarError(RuntimeError):
    """A failure which requires replacing, rather than retrying, this sidecar."""


class UnrecoverableEngineError(UnrecoverableSidecarError):
    """The local inference engine cannot recover inside this replica."""
