"""Auto-load Chidori inference optimizer when CHIDORI_OPTIMIZE_INFERENCE=1.

This is installed as a sitecustomize hook so that `chidori run` can enable
graph decode optimization without requiring the user to add any imports.
"""
import os

if os.environ.get("CHIDORI_OPTIMIZE_INFERENCE") == "1":
    # Use a post-import hook to patch transformers when it's loaded
    import sys

    class _ChidoriImportFinder:
        """Import hook that patches transformers.generate on first import."""

        def find_module(self, fullname, path=None):
            if fullname == "transformers":
                return self
            return None

        def load_module(self, fullname):
            # Remove ourselves so we only fire once
            sys.meta_path.remove(self)
            # Let the real import happen
            import importlib
            module = importlib.import_module(fullname)
            # Now patch
            try:
                from chidori.optimize import optimize
                optimize()
            except Exception:
                pass
            return module

    sys.meta_path.insert(0, _ChidoriImportFinder())
