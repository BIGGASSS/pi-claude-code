import assert from "node:assert/strict";
import test from "node:test";
import type { AssistantMessage, Message, Model, SystemMessage, Tool, ToolResultMessage } from "@earendil-works/pi-ai";
import {
  getInitialSystemMessages,
  makeReplayDiagnostic,
  prepareFrames,
  prepareMessages,
  prepareNativeTranscript,
  TOOL_PREFIX,
} from "../src/transcript.ts";

const model = {
  id: "default",
  api: "claude-code-stream-json",
  provider: "claude-code",
  input: ["text", "image"],
} as unknown as Model<any>;
const usage = {
  input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};
const user = (content = "continue"): Message => ({ role: "user", content, timestamp: 10 });
const system = (content: string): SystemMessage => ({ role: "system", content, timestamp: 2 });
const turn = (ids: string[], overrides: Partial<AssistantMessage> = {}): AssistantMessage => ({
  role: "assistant",
  api: model.api,
  provider: model.provider,
  model: model.id,
  content: ids.map((id) => ({ type: "toolCall", id, name: "read", arguments: { path: id } })),
  usage,
  stopReason: "toolUse",
  timestamp: 1,
  ...overrides,
});
const answer = (id: string, isError = false): ToolResultMessage => ({
  role: "toolResult", toolCallId: id, toolName: "read",
  content: [{ type: "text", text: `answer ${id}` }], isError, timestamp: 3,
});

function deepFreeze(value: unknown): void {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
}

const boundaries = ["user", "assistant", "end"] as const;
function following(boundary: typeof boundaries[number]): Message[] {
  if (boundary === "end") return [];
  if (boundary === "user") return [user()];
  return [turn([], { content: [{ type: "text", text: "next turn" }], stopReason: "stop" }), user()];
}

for (const boundary of boundaries) {
  for (const answered of [[], ["b"], ["c", "a", "b"]]) {
    test(`repairs ${boundary} boundary with ${answered.length} of 3 tool results`, () => {
      const assistant = turn(["a", "b", "c"]);
      // An existing error is still a real result; out-of-order results stay in their original order.
      const realResults = answered.map((id, index) => answer(id, index === 0));
      const next = following(boundary);
      const messages: Message[] = [user("go"), assistant, ...realResults, ...next];
      const original = structuredClone(messages);
      deepFreeze(messages);
      const prepared = prepareMessages(messages, model);
      const missing = ["a", "b", "c"].filter((id) => !answered.includes(id));
      const results = prepared.filter((message) => message.role === "toolResult");

      assert.equal(prepared[1], assistant);
      assert.deepEqual(results.map((result) => result.toolCallId), [...answered, ...missing]);
      realResults.forEach((result, index) => assert.equal(results[index], result));
      assert.deepEqual(results.slice(realResults.length), missing.map((id) => ({
        role: "toolResult", toolCallId: id, toolName: "read",
        content: [{ type: "text", text: "No result provided" }], isError: true, timestamp: assistant.timestamp,
      })));
      assert.deepEqual(prepared.slice(5), next);

      const frames = prepareFrames(prepared);
      const blocks = frames[2]!.message.content as Array<{ type: string; tool_use_id?: string; is_error?: boolean }>;
      assert.equal(frames[2]!.type, "user");
      assert.deepEqual(blocks.slice(0, 3).map((block) => block.tool_use_id), [...answered, ...missing]);
      assert.ok(blocks.slice(0, 3).every((block) => block.type === "tool_result"));
      assert.ok(blocks.slice(realResults.length, 3).every((block) => block.is_error));
      if (boundary === "user") assert.deepEqual(blocks[3], { type: "text", text: "continue" });
      assert.deepEqual(messages, original);
      assert.deepEqual(prepareMessages(messages, model), prepared);
      assert.deepEqual(prepareMessages(prepared, model), prepared);
    });
  }
}

for (const stopReason of ["error", "aborted"] as const) {
  for (const atEnd of [false, true]) {
    test(`omits ${stopReason} assistant including native partial content at ${atEnd ? "history end" : "next user"}`, () => {
      const interrupted = turn(["unfinished"], { stopReason });
      interrupted.content.unshift(
        { type: "thinking", thinking: "partial reasoning", thinkingSignature: "partial signature" },
        { type: "text", text: "partial reply" },
      );
      interrupted.diagnostics = [makeReplayDiagnostic({
        role: "assistant",
        content: [{ type: "tool_use", id: "unfinished", name: `${TOOL_PREFIX}read`, input: {} }],
      }, interrupted) as never];
      const next = atEnd ? [] : [user()];
      const messages: Message[] = [user("go"), interrupted, ...next];
      const original = structuredClone(messages);
      deepFreeze(messages);
      const prepared = prepareMessages(messages, model);

      assert.deepEqual(prepared, [messages[0], ...next]);
      const frames = prepareFrames(prepared);
      assert.deepEqual(frames.map((frame) => frame.type), ["user"]);
      assert.equal(JSON.stringify(frames).includes("unfinished"), false);
      assert.deepEqual(messages, original);
    });
  }
}

for (const boundary of boundaries) {
  test(`holds interleaved system updates after real and synthetic results at ${boundary}`, () => {
    const initial = system("initial");
    const updates: SystemMessage[] = [
      { ...system("first"), sections: { rules: "new rules" } },
      { ...system("second"), toolsRemoved: [{ name: "old" }] },
      system("third"),
    ];
    const assistant = turn(["a", "b", "c"]);
    const realB = answer("b");
    const realA = answer("a", true);
    const next = following(boundary);
    const messages: Message[] = [
      initial, user("go"), assistant, updates[0]!, realB, updates[1]!, realA, updates[2]!, ...next,
    ];
    const original = structuredClone(messages);
    deepFreeze(messages);
    const prepared = prepareMessages(messages, model);

    assert.deepEqual(prepared.slice(0, 5), [initial, messages[1], assistant, realB, realA]);
    assert.equal(prepared[5]!.role, "toolResult");
    assert.equal((prepared[5] as ToolResultMessage).toolCallId, "c");
    assert.deepEqual(prepared.slice(6), [...updates, ...next]);
    updates.forEach((update, index) => assert.equal(prepared[6 + index], update));
    assert.equal(prepared.filter((message) => message.role === "toolResult").length, 3);
    const frames = prepareFrames(prepared);
    assert.deepEqual(frames[2]!.message.content.slice(0, 3).map((block) => (block as { tool_use_id: string }).tool_use_id), ["b", "a", "c"]);
    assert.deepEqual(messages, original);
    assert.deepEqual(prepareMessages(prepared, model), prepared);
  });
}

test("does not duplicate a completed result across system messages", () => {
  const assistant = turn(["a", "b"]);
  const first = answer("a");
  const second = answer("b", true);
  const updates = [system("before"), system("between"), system("after")];
  const messages: Message[] = [user(), assistant, updates[0]!, first, updates[1]!, second, updates[2]!];
  const prepared = prepareMessages(messages, model);
  assert.deepEqual(prepared, [messages[0], assistant, first, second, ...updates]);
});

test("closes each tool batch before the next assistant, even when that assistant is skipped", () => {
  const first = turn(["a", "b"]);
  const last = turn(["c", "d"], { stopReason: "length" });
  const update = system("held update");
  const afterError = system("after error");
  const messages: Message[] = [
    user("go"), first, answer("a"), update,
    turn(["discarded"], { stopReason: "error" }), afterError,
    last, answer("d"), user(),
  ];
  const prepared = prepareMessages(messages, model);
  assert.deepEqual(prepared.map((message) => message.role), [
    "user", "assistant", "toolResult", "toolResult", "system", "system",
    "assistant", "toolResult", "toolResult", "user",
  ]);
  assert.deepEqual(prepared.filter((message) => message.role === "toolResult").map((result) => result.toolCallId), ["a", "b", "d", "c"]);
  assert.equal(prepared[4], update);
  assert.equal(prepared[5], afterError);
  assert.equal(prepared[6], last);
  const frames = prepareFrames(prepared);
  assert.deepEqual(frames.map((frame) => frame.type), ["user", "assistant", "user", "assistant", "user"]);
});

test("accounts for both existing and missing results using normalized foreign IDs", () => {
  const ids = ["call|answered_", "call|missing_"].map((prefix) => prefix + "x".repeat(400));
  const normalizedIds = ids.map((id) => id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64));
  const foreign = turn(ids, { api: "openai-responses", provider: "openai", model: "gpt-5" });
  foreign.content.unshift({ type: "thinking", thinking: "foreign reasoning", thinkingSignature: "foreign signature" });
  foreign.diagnostics = [makeReplayDiagnostic({
    role: "assistant", content: [{ type: "thinking", thinking: "stale", signature: "stale signature" }],
  }, foreign) as never];
  const real = answer(ids[0]!);
  const update = system("update");
  const messages: Message[] = [user(), foreign, update, real];
  const original = structuredClone(messages);
  deepFreeze(messages);
  const prepared = prepareMessages(messages, model);
  const assistant = prepared[1] as AssistantMessage;
  const results = prepared.filter((message) => message.role === "toolResult");

  assert.deepEqual(results.map((result) => result.toolCallId), normalizedIds);
  assert.deepEqual(assistant.content.map((block) => block.type), ["toolCall", "toolCall"]);
  assert.deepEqual(assistant.content.map((block) => (block as { id: string }).id), normalizedIds);
  assert.deepEqual(assistant.diagnostics, []);
  assert.deepEqual(results[0], { ...real, toolCallId: normalizedIds[0] });
  assert.equal(results[1]!.isError, true);
  assert.equal(prepared.at(-1), update);
  const frames = prepareFrames(prepared);
  assert.deepEqual(frames[2]!.message.content.map((block) => (block as { tool_use_id: string }).tool_use_id), normalizedIds);
  assert.deepEqual(messages, original);
  assert.deepEqual(prepareMessages(messages, model), prepared);
  assert.deepEqual(prepareMessages(prepared, model), prepared);
});

test("repairs missing results without altering native signed reasoning, response model, or system metadata", () => {
  const read = { name: "read", description: "read", parameters: { type: "object", properties: {} } } as unknown as Tool;
  const initial: SystemMessage = { ...system("initial"), toolsAdded: [read] };
  const update: SystemMessage = { ...system("updated"), sections: { rules: "keep this" }, toolsRemoved: [{ name: "read" }] };
  const assistant = turn(["a", "b"], {
    responseModel: "claude-opus-5-5", responseId: "response-1", providerThinkingLevel: "high",
  });
  assistant.content.unshift(
    { type: "thinking", thinking: "private", thinkingSignature: "signed" },
    { type: "thinking", thinking: "[Reasoning redacted]", thinkingSignature: "opaque", redacted: true },
    { type: "text", text: "checking" },
  );
  const native = {
    id: "native-1", role: "assistant" as const, model: "claude-opus-5-5",
    content: [
      { type: "thinking" as const, thinking: "private", signature: "signed", extra: "preserved" },
      { type: "redacted_thinking" as const, data: "opaque" },
      { type: "text" as const, text: "checking" },
      { type: "tool_use" as const, id: "a", name: `${TOOL_PREFIX}read`, input: { path: "a" } },
      { type: "tool_use" as const, id: "b", name: `${TOOL_PREFIX}read`, input: { path: "b" } },
    ],
  };
  const initialSystems = [{ role: "system", content: [{ type: "text", text: "native environment" }] }];
  assistant.diagnostics = [makeReplayDiagnostic(native, assistant, initialSystems) as never];
  const messages: Message[] = [initial, user(), assistant, update, answer("a"), user()];
  const original = structuredClone(messages);
  deepFreeze(messages);
  const prepared = prepareMessages(messages, model);
  const frames = prepareFrames(prepared);

  assert.equal(prepared[2], assistant);
  assert.equal((prepared[2] as AssistantMessage).diagnostics, assistant.diagnostics);
  assert.equal(prepared[5], update);
  assert.equal(frames[1]!.message.model, "claude-opus-5-5");
  assert.deepEqual(frames[1]!.message.content, native.content);
  assert.deepEqual(frames[2]!.message.content, [
    { type: "tool_result", tool_use_id: "a", content: "answer a", is_error: false },
    { type: "tool_result", tool_use_id: "b", content: "No result provided", is_error: true },
    { type: "text", text: "continue" },
  ]);
  assert.deepEqual(getInitialSystemMessages(prepared), initialSystems);
  assert.deepEqual(prepareNativeTranscript(prepared, true), prepareNativeTranscript(messages, true));
  assert.deepEqual(messages, original);
  assert.deepEqual(prepareMessages(messages, model), prepared);
  assert.deepEqual(prepareMessages(prepared, model), prepared);
});

test("leaves histories with no pending calls unchanged", () => {
  const messages: Message[] = [
    system("initial"), user(),
    turn([], { content: [{ type: "text", text: "done" }], stopReason: "stop" }),
    system("update"), user(),
  ];
  deepFreeze(messages);
  const prepared = prepareMessages(messages, model);
  assert.notEqual(prepared, messages);
  assert.deepEqual(prepared, messages);
  messages.forEach((message, index) => assert.equal(prepared[index], message));
});
