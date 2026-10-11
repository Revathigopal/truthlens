"""Exercise the real HTTP client against a local server that speaks the chat-completions protocol."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from truthlens.llm import LLMError, OpenAICompatLLM, get_llm
from truthlens.mock import mock_handler
from truthlens.pipeline import TruthLensPipeline

SEEN = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "auth": self.headers.get("Authorization"), "model": body["model"]})
        if body["model"] == "broken":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"model crashed")
            return
        reply = {"choices": [{"message": {"role": "assistant", "content": mock_handler(body["messages"])}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def base_url():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_pipeline_runs_over_http(base_url, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", base_url)
    SEEN.clear()
    llm = get_llm("ollama", model="test-model")
    report = TruthLensPipeline(llm).verify("Sales doubled in 2025. It is the best product.", source="brand advert")
    assert len(SEEN) == 6  # one request per agent
    assert all(call["path"] == "/v1/chat/completions" and call["model"] == "test-model" for call in SEEN)
    assert report.source_profile.kind == "interested_party"


def test_api_key_is_sent_as_bearer_token(base_url):
    SEEN.clear()
    llm = OpenAICompatLLM(base_url, "m", api_key="secret")
    llm.complete([{"role": "system", "content": "AGENT:claim_extractor"}, {"role": "user", "content": '{"text": "Water boils."}'}])
    assert SEEN[0]["auth"] == "Bearer secret"


def test_http_errors_and_unreachable_hosts_raise_llm_error(base_url):
    with pytest.raises(LLMError, match="HTTP 500"):
        OpenAICompatLLM(base_url, "broken").complete([{"role": "user", "content": "x"}])
    with pytest.raises(LLMError, match="could not reach"):
        OpenAICompatLLM("http://127.0.0.1:9/v1", "m", timeout=2).complete([{"role": "user", "content": "x"}])


def test_missing_key_and_unknown_provider(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        get_llm("groq")
    with pytest.raises(LLMError, match="unknown provider"):
        get_llm("mystery")
