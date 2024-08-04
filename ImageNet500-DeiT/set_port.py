import os
from torch.distributed import TCPStore

port = int(os.environ.get('MY_APP_PORT', 12345))

self._store = TCPStore("127.0.0.1", port)