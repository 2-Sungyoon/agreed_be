import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from infra.llm.harness import (
    AgentPolicy,
    AgentTool,
    StopReason,
    ToolRisk,
    run_agent_with_trace,
)


class Answer(BaseModel):
    answer: str


def _tool_call(call_id: str, name: str, arguments: dict | str):
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=raw),
    )


def _response(*, tool_calls=None, content=""):
    message = SimpleNamespace(tool_calls=tool_calls or [], content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("준비된 모델 응답이 없습니다.")
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=completions)
        self.requests = completions.requests


class AgentHarnessTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, *, responses, tools, policy=AgentPolicy(), approval_callback=None):
        client = FakeClient(responses)
        with (
            patch("infra.llm.harness.has_api_key", return_value=True),
            patch("infra.llm.harness.get_client", return_value=client),
        ):
            outcome = await run_agent_with_trace(
                system_prompt="근거를 확인한다.",
                task="외부 요청 원문",
                tools=tools,
                schema=Answer,
                policy=policy,
                approval_callback=approval_callback,
            )
        return outcome, client

    async def test_duplicate_tool_call_is_blocked_and_trace_has_no_raw_data(self):
        executed = []

        async def search(arguments):
            executed.append(arguments)
            return "SECRET-TOOL-RESULT"

        tool = AgentTool(
            name="search",
            description="자료를 검색한다.",
            parameters={
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
                "additionalProperties": False,
            },
            run=search,
        )
        responses = [
            _response(tool_calls=[_tool_call("1", "search", {"keyword": "계약"})]),
            _response(tool_calls=[_tool_call("2", "search", {"keyword": "계약"})]),
            _response(),
            _response(content='{"answer":"확인 완료"}'),
        ]

        outcome, client = await self._run(responses=responses, tools=[tool])

        self.assertEqual([{"keyword": "계약"}], executed)
        self.assertEqual("확인 완료", outcome.value.answer)
        self.assertEqual(
            ["ok", "duplicate_blocked"],
            [item.status for item in outcome.trace.tool_calls],
        )
        self.assertNotIn("SECRET-TOOL-RESULT", str(outcome.trace.safe_summary()))
        self.assertIn("<untrusted_tool_data>", str(client.requests[1]["messages"]))

    async def test_write_tool_fails_closed_without_human_approval(self):
        executed = []

        async def update(arguments):
            executed.append(arguments)
            return "변경 완료"

        tool = AgentTool(
            name="update_contract",
            description="계약을 변경한다.",
            parameters={"type": "object", "properties": {}},
            run=update,
            risk=ToolRisk.WRITE,
        )
        responses = [
            _response(tool_calls=[_tool_call("1", "update_contract", {})]),
            _response(),
            _response(content='{"answer":"승인 필요"}'),
        ]

        outcome, _ = await self._run(responses=responses, tools=[tool])

        self.assertEqual([], executed)
        self.assertEqual("approval_required", outcome.trace.tool_calls[0].status)
        self.assertEqual("승인 필요", outcome.value.answer)

    async def test_invalid_arguments_do_not_reach_tool(self):
        executed = []

        async def read(arguments):
            executed.append(arguments)
            return "읽음"

        tool = AgentTool(
            name="read_file",
            description="파일을 읽는다.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            run=read,
        )
        responses = [
            _response(tool_calls=[_tool_call("1", "read_file", {"path": 7, "extra": True})]),
            _response(),
            _response(content='{"answer":"확인 불가"}'),
        ]

        outcome, _ = await self._run(responses=responses, tools=[tool])

        self.assertEqual([], executed)
        self.assertEqual("invalid_arguments", outcome.trace.tool_calls[0].status)

    async def test_tool_budget_forces_bounded_conclusion(self):
        async def read(_):
            return "근거"

        tool = AgentTool(
            name="read_contract",
            description="계약을 읽는다.",
            parameters={"type": "object", "properties": {}},
            run=read,
        )
        responses = [
            _response(tool_calls=[_tool_call("1", "read_contract", {})]),
            _response(content='{"answer":"예산 내 결론"}'),
        ]
        policy = AgentPolicy(max_total_tool_calls=1)

        outcome, _ = await self._run(responses=responses, tools=[tool], policy=policy)

        self.assertEqual(StopReason.TOOL_BUDGET, outcome.trace.stop_reason)
        self.assertEqual("예산 내 결론", outcome.value.answer)
        self.assertEqual(1, len(outcome.trace.tool_calls))

    async def test_approved_write_tool_runs_once(self):
        executed = []

        async def update(arguments):
            executed.append(arguments)
            return "변경 완료"

        async def approve(tool, arguments):
            return tool.name == "update" and arguments == {"id": "A-1"}

        tool = AgentTool(
            name="update",
            description="승인된 값을 변경한다.",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            run=update,
            risk=ToolRisk.WRITE,
        )
        responses = [
            _response(tool_calls=[_tool_call("1", "update", {"id": "A-1"})]),
            _response(),
            _response(content='{"answer":"반영됨"}'),
        ]

        outcome, _ = await self._run(
            responses=responses,
            tools=[tool],
            approval_callback=approve,
        )

        self.assertEqual([{"id": "A-1"}], executed)
        self.assertEqual("ok", outcome.trace.tool_calls[0].status)


if __name__ == "__main__":
    unittest.main()
