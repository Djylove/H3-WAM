from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

try:
    import msgpack
    import msgpack_numpy as msgpack_numpy
    import zmq
except ImportError as exc:
    raise ImportError(
        "AnyGrasp policy server requires `msgpack`, `msgpack-numpy`, and `pyzmq`. "
        "Install the updated project dependencies, e.g. `pip install -e .`, "
        "or install `msgpack==1.1.0 msgpack-numpy==0.4.8 pyzmq==27.0.1`."
    ) from exc


class MsgpackNumpySerializer:
    """Small msgpack serializer for numpy-heavy policy RPC payloads."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=msgpack_numpy.encode, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)


@dataclass
class EndpointHandler:
    handler: Callable[..., Any]
    requires_input: bool = True


class PolicyServer:
    """ZMQ REP server compatible with the GR00T-style policy endpoint contract."""

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 5555,
        api_token: Optional[str] = None,
    ) -> None:
        self.policy = policy
        self.host = str(host)
        self.port = int(port)
        self.api_token = api_token
        self.running = True
        self._closed = False
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        self._endpoints: dict[str, EndpointHandler] = {}

        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._handle_kill, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint("get_modality_config", self.policy.get_modality_config, requires_input=False)

    def register_endpoint(self, name: str, handler: Callable[..., Any], *, requires_input: bool = True) -> None:
        self._endpoints[str(name)] = EndpointHandler(handler=handler, requires_input=requires_input)

    def _handle_ping(self) -> dict[str, Any]:
        return {"status": "ok", "message": "FastWAM AnyGrasp policy server is running"}

    def _handle_kill(self) -> dict[str, Any]:
        self.running = False
        return {"status": "ok", "message": "server stopping"}

    def _validate_token(self, request: dict[str, Any]) -> bool:
        if self.api_token is None:
            return True
        return request.get("api_token") == self.api_token

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.running = False
        self.socket.close(linger=0)
        self.context.term()

    def run(self) -> None:
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"FastWAM AnyGrasp server is ready on {addr}", flush=True)
        while self.running:
            try:
                request = MsgpackNumpySerializer.from_bytes(self.socket.recv())
                if not isinstance(request, dict):
                    raise TypeError(f"RPC request must be a dict, got {type(request)}")
                if not self._validate_token(request):
                    self.socket.send(MsgpackNumpySerializer.to_bytes({"error": "Unauthorized"}))
                    continue

                endpoint = str(request.get("endpoint", "get_action"))
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                endpoint_handler = self._endpoints[endpoint]
                if endpoint_handler.requires_input:
                    data = request.get("data", {})
                    if data is None:
                        data = {}
                    if not isinstance(data, dict):
                        raise TypeError(f"Endpoint data must be a dict, got {type(data)}")
                    result = endpoint_handler.handler(**data)
                else:
                    result = endpoint_handler.handler()
                self.socket.send(MsgpackNumpySerializer.to_bytes(result))
            except Exception as exc:
                import traceback

                print(f"Error in FastWAM AnyGrasp server: {exc}", flush=True)
                print(traceback.format_exc(), flush=True)
                self.socket.send(MsgpackNumpySerializer.to_bytes({"error": str(exc)}))

    def __enter__(self) -> "PolicyServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PolicyClient:
    """Lightweight client with the same endpoints as the server."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: Optional[str] = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.api_token = api_token
        self.context = zmq.Context()
        self._closed = False
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: Optional[dict[str, Any]] = None, *, requires_input: bool = True) -> Any:
        request: dict[str, Any] = {"endpoint": str(endpoint)}
        if requires_input:
            request["data"] = {} if data is None else data
        if self.api_token is not None:
            request["api_token"] = self.api_token
        try:
            self.socket.send(MsgpackNumpySerializer.to_bytes(request))
            response = MsgpackNumpySerializer.from_bytes(self.socket.recv())
        except zmq.error.Again:
            self.socket.close(linger=0)
            self._init_socket()
            raise
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self.socket.close(linger=0)
            self._init_socket()
            return False

    def get_action(
        self,
        observation: dict[str, Any],
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": options})
        return tuple(response)  # msgpack serializes tuples as arrays

    def reset(self, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, Any]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def kill_server(self) -> dict[str, Any]:
        return self.call_endpoint("kill", requires_input=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.socket.close(linger=0)
        self.context.term()

    def __enter__(self) -> "PolicyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
