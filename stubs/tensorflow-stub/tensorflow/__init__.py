# Intentionally near-empty stub package named `tensorflow`.
#
# Molmo's trust_remote_code loader statically scans its remote-code files for every
# top-level `import` statement and refuses to load unless each one is importable -
# even though the only `import tensorflow` in that code is inside a dead branch
# (`resize_method == "tensorflow"`, never true for our default config). This package
# exists solely so `importlib.import_module("tensorflow")` succeeds during that check.
#
# einops' backend auto-detection (used elsewhere in Molmo's code, on real numpy/torch
# tensors) also unconditionally probes every known framework via
# `isinstance(tensor, (tf.Tensor, tf.Variable))`, so those two names have to exist -
# as classes nothing real will ever be an instance of, so the probe safely returns
# False. Beyond that, this stub implements nothing: if Molmo's actual dead
# tensorflow-resize branch were ever reached, it should fail loudly (AttributeError)
# rather than silently behave incorrectly.


class Tensor:
    pass


class Variable:
    pass
