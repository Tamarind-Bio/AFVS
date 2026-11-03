"""Loaded (via PYTHONPATH) into every interpreter a local-mode test starts:
the AFVS scripts, afvs_run.py and its multiprocessing children.

The local backend promises no boto3 import and no network, so both fail loudly
here: importing boto3/botocore raises, and so does any inet connect."""

import socket
import sys

for _name in ("boto3", "botocore"):
    sys.modules[_name] = None

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def _guard(sock):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        raise OSError("network access attempted in a local-mode path (blocked by the test guard)")


def connect(self, address):
    _guard(self)
    return _connect(self, address)


def connect_ex(self, address):
    _guard(self)
    return _connect_ex(self, address)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
