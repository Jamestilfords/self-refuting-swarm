"""Offline contract tests: providers and Modal are replaced, never contacted."""
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skeptic import agents, api
from skeptic.agent_runtime import BudgetExhausted, RunBudget
from skeptic.memory import LocalMemory, PastDevMemory
from skeptic.streams.common import render
from skeptic.streams.proposer import Proposer, smoke_ctx
from skeptic.streams.scout import Scout


def budget(path, **kwargs):
    return RunBudget(path, max_calls=kwargs.get("max_calls", 2), max_searches=1,
                     max_tokens=kwargs.get("max_tokens", 10_000), max_output_tokens=100)


def test_shared_budget_survives_failure_and_resume(tmp_path):
    state = tmp_path / "budget.json"
    b = budget(state)
    def failed(*_):
        raise RuntimeError("provider failed")
    with pytest.raises(RuntimeError):
        b.llm(failed)("system", "user")
    reserved = b.used["charged_tokens"]
    assert reserved > 0 and b.used["calls"] == 1
    b.search()
    resumed = budget(state)
    assert resumed.used == b.used
    with pytest.raises(BudgetExhausted):
        resumed.search()
    resumed.llm(lambda *_: ("answer", {"input_tokens": 5, "output_tokens": 3}))("s", "u")
    assert resumed.used["charged_tokens"] == reserved + 8
    with pytest.raises(BudgetExhausted):
        resumed.llm(failed)("s", "u")
    with pytest.raises(ValueError, match="original run limits"):
        budget(state, max_calls=3)


def test_budget_stops_before_call_and_unknown_usage_stays_reserved(tmp_path):
    b = budget(tmp_path / "small.json", max_tokens=100)
    with pytest.raises(BudgetExhausted):
        b.llm(lambda *_: pytest.fail("called beyond budget"))("s", "u")
    b = budget(tmp_path / "missing.json")
    b.llm(lambda *_: ("text", {"input_tokens": None, "output_tokens": None}))("s", "u")
    assert b.used["charged_tokens"] == 1126


def test_scout_bounds_sources_and_persists_cursor(tmp_path):
    memory = LocalMemory(tmp_path / "memory.jsonl")
    class Exa:
        def search(self, *args, **kwargs):
            return SimpleNamespace(results=[
                SimpleNamespace(url="https://example.org/title-only", text=""),
                SimpleNamespace(url="https://user:secret@example.org/a", text="Research"),
                SimpleNamespace(url="https://example.org/paper", title="Paper", text="kNN method"),
                SimpleNamespace(url="https://example.org/extra", text="must be ignored"),
            ])
    prompts = []
    def llm(system, user):
        prompts.append((system, user))
        return "Use the k-th training neighbor distance.", {"input_tokens": 3, "output_tokens": 2}
    state = tmp_path / "scout.json"
    b = budget(tmp_path / "budget.json")
    scout = Scout(memory, b.llm(llm), Exa(), state_path=state, before_search=b.search)
    assert scout.search() == ["m_0001"]
    assert len(prompts) == 1 and "untrusted" in prompts[0][0]
    content = memory.get("m_0001")["content"]
    assert len(content["excerpt_sha256"]) == 64 and content["query"]
    resumed = Scout(memory, llm, Exa(), state_path=state)
    assert resumed._round == 1 and resumed.search() == []
    assert b.used["searches"] == 1 and b.used["calls"] == 1


def test_proposer_never_executes_generated_or_resumed_source(tmp_path, monkeypatch):
    ctx = smoke_ctx()
    source = render(family="knn", arm="ce", rationale="Distance", body="raise RuntimeError('never execute here')")
    monkeypatch.setattr(api, "load_proposal", lambda *_: pytest.fail("host execution"))
    seen = []
    def validator(text):
        seen.append(text)
        return np.arange(len(ctx.eval), dtype=float)
    mem = LocalMemory(tmp_path / "memory.jsonl")
    p = Proposer(mem, lambda *_: (source, {}), ctx, tmp_path / "proposals", validator=validator)
    assert p.propose_one(0)["status"] == "ok"
    p.save()
    again = Proposer(mem, None, ctx, p.out, validator=validator)
    assert len(seen) == 2 and again.usage["calls"] == 1
    (p.out / f"{p.rows[0]['sha']}.py").write_text(source + "# changed\n")
    with pytest.raises(ValueError, match="no longer match"):
        Proposer(mem, None, ctx, p.out, validator=validator)


def test_provider_exception_bodies_do_not_enter_logs(tmp_path, capsys):
    secret = "sensitive-provider-value"
    def failed(*_):
        raise RuntimeError(secret)
    mem = LocalMemory(tmp_path / "memory.jsonl")
    p = Proposer(mem, failed, smoke_ctx(), tmp_path / "out", validator=failed)
    p.run(1, scout=SimpleNamespace(search=failed))
    assert secret not in capsys.readouterr().out
    assert secret not in (p.out / "llm.json").read_text()
    class Down(PastDevMemory):
        def _call(self, *args, **kwargs):
            import urllib.error
            raise urllib.error.URLError(secret)
    remote = Down(tmp_path / "remote.jsonl", api_key="fake")
    remote.write("lesson", "analyst", {"text": "Local evidence"})
    assert secret not in capsys.readouterr().out


def test_pastdev_namespaces_and_filters_foreign_or_sealed_recall(tmp_path):
    bodies = []
    class Fake(PastDevMemory):
        def _call(self, method, path, body=None):
            bodies.append(body)
            if path == "/recall":
                return {"results": [
                    {"content": "[other:m_0001] [m_0001] Foreign evidence"},
                    {"content": "[this-run:m_0001] [m_0001] Local evidence"},
                    {"content": "[this-run:m_0001] [m_0001] OOS_TEST scores"},
                ]}
            return {"status": "queued"}
    memory = Fake(tmp_path / "memory.jsonl", api_key="fake", namespace="this-run")
    memory.write("lesson", "analyst", {"text": "Local evidence"})
    assert bodies[0]["id"] == "this-run:m_0001"
    recalled = memory.recall("evidence")
    assert len(recalled) == 1 and recalled[0]["ids"] == ["m_0001"]


def test_isolated_validation_uses_fresh_full_repeat_and_subset(monkeypatch):
    from modal_jobs.agent_validation import validate_isolated
    from skeptic.sandbox import unpack_input
    ctx = smoke_ctx()
    source = render(family="other", arm="ce", rationale="Fixture", body="raise RuntimeError()")
    monkeypatch.setattr(api, "load_proposal", lambda *_: pytest.fail("host execution"))
    calls = []
    def scorer(text, sha, payload, n, k, **kwargs):
        calls.append(kwargs)
        c = unpack_input(payload)
        assert c.eval.labels is None
        return c.eval.logits[:, 0], None, 0.1
    assert np.array_equal(validate_isolated(source, ctx, scorer), ctx.eval.logits[:, 0])
    assert calls == [{"return_probs": True}, {"return_probs": True}, {"subset": True, "return_probs": True}]
    def dependent(text, sha, payload, n, k, **kwargs):
        c = unpack_input(payload)
        values = c.eval.logits[:, 0]
        return values - values.mean(), None, 0.1
    with pytest.raises(api.Malformed, match="subset_dependent"):
        validate_isolated(source, ctx, dependent)
    counter = 0
    def nondeterministic(text, sha, payload, n, k, **kwargs):
        nonlocal counter
        counter += 1
        return np.arange(n) + counter, None, 0.1
    with pytest.raises(api.Malformed, match="nondeterministic"):
        validate_isolated(source, ctx, nondeterministic)


def test_modal_validator_registers_worker_before_entering_context(monkeypatch):
    """Modal rejects @app.function defined inside modal_validator at runtime."""
    import modal_jobs.agent_validation as validation

    assert "image_ready" in validation.agent_app._local_state.functions
    assert validation.image_ready._raw_f_.__qualname__ == "image_ready"
    monkeypatch.setattr(validation.agent_app, "run", lambda **_: nullcontext())
    monkeypatch.setattr(validation.modal, "enable_output", nullcontext)
    with validation.modal_validator(smoke_ctx()) as validate:
        assert callable(validate)


def test_environment_never_overrides_shell_or_interpolates(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    monkeypatch.setenv("EXA_API_KEY", "from-shell")
    monkeypatch.delenv("PASTDEV_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("EXA_API_KEY=from-file\nPASTDEV_API_KEY=${EXA_API_KEY}\n")
    agents.load_environment(path)
    assert os.environ["EXA_API_KEY"] == "from-shell"
    assert os.environ["PASTDEV_API_KEY"] == "${EXA_API_KEY}"


def test_cli_presence_check_is_offline_and_blocks_incomplete_modal_pair(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agents, "load_environment", lambda *_: None)
    for key in agents.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MODAL_TOKEN_ID", "fake-id")
    monkeypatch.setattr(agents, "run", lambda *_: pytest.fail("must fail before making API calls"))
    out = tmp_path / "never-created"
    assert agents.main(["plan", "--run-dir", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "MODAL_TOKEN_SECRET" in report["readiness"]["missing"]
    assert agents.main(["run", "--run-dir", str(out)]) == 2
    assert not out.exists() and "fake-id" not in capsys.readouterr().out


def test_cli_workflow_and_resume_without_live_services(tmp_path, monkeypatch, capsys):
    exa = pytest.importorskip("exa_py")
    import modal_jobs.agent_validation as validation
    import skeptic.streams.proposer as proposer
    monkeypatch.setattr(agents, "load_environment", lambda *_: None)
    monkeypatch.setattr(agents, "readiness", lambda *_: {"ready": True})
    monkeypatch.setenv("EXA_API_KEY", "fake")
    monkeypatch.setattr(exa, "Exa", lambda **_: SimpleNamespace(search=lambda *a, **k: SimpleNamespace(results=[
        SimpleNamespace(url="https://example.org/paper", title="Paper", text="A scoring method")
    ])))
    source = render(family="knn", arm="ce", rationale="Fixture", body="raise RuntimeError('isolated')",
                    source="https://example.org/paper", memory_used=["m_0001"])
    responses = iter(["Distance to neighbors", source])
    monkeypatch.setattr(proposer, "openai_llm", lambda *a, **k: lambda *_: (
        next(responses), {"input_tokens": 10, "output_tokens": 5}))
    @contextmanager
    def validator(ctx):
        yield lambda _: np.arange(len(ctx.eval), dtype=float)
    monkeypatch.setattr(validation, "modal_validator", validator)
    args = ["run", "--run-dir", str(tmp_path / "run"), "--model", "fake", "--memory-backend", "local", "--n", "1"]
    assert agents.main(args) == 0
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["accepted"] == 1 and report["usage"]["calls"] == 2 and report["usage"]["searches"] == 1
    monkeypatch.setattr(proposer, "openai_llm", lambda *a, **k: pytest.fail("already complete"))
    assert agents.main(args) == 0
    assert "Target already reached" in capsys.readouterr().out
    (tmp_path / "run/budget.json").unlink()
    assert agents.main(args) == 2
    assert "Saved budget is missing" in capsys.readouterr().out


def test_run_lock_rejects_a_second_writer(tmp_path):
    with agents.run_lock(tmp_path):
        with pytest.raises(ValueError, match="Another agent process"):
            with agents.run_lock(tmp_path):
                pytest.fail("a second writer acquired the same run")


def test_locked_exa_sdk_request_shape_without_network(tmp_path, monkeypatch):
    exa = pytest.importorskip("exa_py")
    client = exa.Exa(api_key="test-key", timeout=30)
    requests = []
    def request(endpoint, data, **kwargs):
        requests.append((endpoint, data))
        return {"requestId": "fixture", "results": [
            {"id": "fixture", "url": "https://example.org/paper", "title": "Paper", "text": "kNN scoring"}
        ]}
    monkeypatch.setattr(client, "request", request)
    scout = Scout(LocalMemory(tmp_path / "memory.jsonl"), lambda *_: ("kNN distance", {}), client)
    assert scout.search() == ["m_0001"]
    assert len(requests) == 1 and requests[0][0] == "/search"
    assert requests[0][1]["numResults"] == 3
    assert requests[0][1]["contents"]["text"]["maxCharacters"] == 2500


def test_openai_adapter_sets_output_timeout_and_disables_retries(monkeypatch):
    openai = pytest.importorskip("openai")
    from skeptic.streams.proposer import openai_llm
    created, requests = [], []
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_text="source", usage=SimpleNamespace(input_tokens=12, output_tokens=5))
    def client(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(openai, "OpenAI", client)
    text, usage = openai_llm("test-model", max_output_tokens=512)("system", "prompt")
    assert text == "source" and usage == {"input_tokens": 12, "output_tokens": 5}
    assert created == [{"timeout": 45, "max_retries": 0}]
    assert requests[0]["max_output_tokens"] == 512 and requests[0]["store"] is False
