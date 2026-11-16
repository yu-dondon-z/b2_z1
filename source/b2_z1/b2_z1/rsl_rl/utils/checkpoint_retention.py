"""Bound storage for checkpoints written by this runner in its own run directory."""

from pathlib import Path


class CheckpointRetention:
    """Never scan old runs or remove files that predate this runner instance."""

    def __init__(self, log_dir, keep_last=0):
        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 0:
            raise ValueError("keep_last_checkpoints must be a nonnegative integer")
        self.log_dir = Path(log_dir).resolve() if log_dir is not None else None
        self.keep_last = keep_last
        self._saved = []

    def eligible(self, path):
        path = Path(path)
        return (
            self.log_dir is not None
            and not path.is_symlink()
            and path.resolve().parent == self.log_dir
            and path.name.startswith("model_")
            and path.name.endswith(".pt")
            and path.name[6:-3].isdigit()
        )

    def saved(self, path, existed_before=False):
        """Call only after a successful save; retain pre-existing files forever."""
        if not self.keep_last or existed_before or not self.eligible(path):
            return
        path = Path(path).resolve()
        if path in self._saved:
            self._saved.remove(path)
        self._saved.append(path)
        while len(self._saved) > self.keep_last:
            old = self._saved.pop(0)
            # Recheck scope and symlinks before removal, even for tracked paths.
            if self.eligible(old):
                old.unlink(missing_ok=True)
