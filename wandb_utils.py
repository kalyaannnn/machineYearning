class _NoOpWandbRun:
    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


class _NoOpWandbModule:
    def init(self, *args, **kwargs):
        print("wandb not installed; proceeding with logging disabled.")
        return _NoOpWandbRun()

    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


def get_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except ImportError:
        return _NoOpWandbModule()
