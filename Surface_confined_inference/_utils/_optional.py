class MissingDependency:
    """Stand-in for an optional import that failed; raises on any use."""

    def __init__(self, name, hint):
        self._name = name
        self._hint = hint

    def _raise(self):
        raise ImportError(f"{self._name} is not available. {self._hint}")

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        self._raise()

    def __call__(self, *args, **kwargs):
        self._raise()

    def __bool__(self):
        return False


AX_HINT = "Install the optional Ax dependencies with `pip install -e .[ax]`."
C_HINT = "The C extension was not built; reinstall without SCI_NO_COMPILE set."
