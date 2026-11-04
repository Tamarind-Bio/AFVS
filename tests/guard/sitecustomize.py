"""Loaded (via PYTHONPATH) into every interpreter a docking-job test starts:
afvs_run.py and its multiprocessing children.

A presigned docking job has no AWS identity and must not import boto3 or reach
anything but its presigned URLs, so here importing boto3/botocore raises, and
so does any inet connect except to loopback (the fake S3) when allowed."""

import os
import socket
import sys

for _name in ("boto3", "botocore"):
    sys.modules[_name] = None

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


# Presigned-mode tests talk to a fake presigned-URL server on loopback; they
# set AFVS_TEST_ALLOW_LOOPBACK=1 (boto3 stays blocked either way)
_ALLOW_LOOPBACK = os.environ.get("AFVS_TEST_ALLOW_LOOPBACK") == "1"


def _guard(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        if _ALLOW_LOOPBACK and isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1", "localhost"):
            return
        raise OSError("network access outside loopback (blocked by the test guard)")


def connect(self, address):
    _guard(self, address)
    return _connect(self, address)


def connect_ex(self, address):
    _guard(self, address)
    return _connect_ex(self, address)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
