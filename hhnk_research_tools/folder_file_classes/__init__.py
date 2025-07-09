import lazy_loader as _lazy
import time

t0 = time.time()

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)


print(f"Load in {(time.time() - t0):.3f}s folder_file_classes.__init__")
