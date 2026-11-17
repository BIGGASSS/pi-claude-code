import assert from "node:assert/strict";
import * as http from "node:http";
import { chmod, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { isRetryableAssistantError, type AssistantMessage } from "@earendil-works/pi-ai";
import {
  cachedClaudeProcessCount,
  closeClaudeProcesses,
  discoverClaudeModels,
  getClaudeStatus,
  rewriteInitialRequest,
  runClaudeRequest,
} from "../src/claude-cli.ts";
import type { NativeFrame } from "../src/native-types.ts";
import { createClaudeCodeProvider } from "../src/provider.ts";
import { getInitialSystemMessages, makeReplayDiagnostic, prepareFrames } from "../src/transcript.ts";

type Block = Record<string, unknown>;
type WireMessage = { role: string; content: unknown };

const text = (value: string): Block => ({ type: "text", text: value });
const memory = "<system-reminder>[MEMORY.md contents]</system-reminder>";
const account = "<system-reminder># userEmail\nuser@example.com</system-reminder>";

/** Canonical frames for the non-system messages of a wire request without reminders. */
function framesFor(messages: WireMessage[]): NativeFrame[] {
  return messages
    .filter((message) => message.role !== "system")
    .map((message) => ({
      type: message.role as "user" | "assistant",
      message: { role: message.role as "user" | "assistant", content: structuredClone(message.content) as unknown[] },
    }));
}

function rewrite(
  wire: WireMessage[],
  frames: NativeFrame[],
  initialSystemMessages?: unknown[],
  injections?: Parameters<typeof rewriteInitialRequest>[3],
): WireMessage[] {
  const body = Buffer.from(JSON.stringify({ messages: wire }));
  return (JSON.parse(rewriteInitialRequest(body, frames, initialSystemMessages, injections).toString("utf8")) as {
    messages: WireMessage[];
  }).messages;
}

const withoutMarkers = (value: unknown): unknown => JSON.parse(JSON.stringify(value, (key, child) => (key === "cache_control" ? undefined : child)));

/** Number of cache_control markers anywhere in a value. */
const count = (value: unknown): number =>
  Array.isArray(value)
    ? value.reduce((sum: number, child) => sum + count(child), 0)
    : value && typeof value === "object"
      ? Object.entries(value).reduce(
          (sum: number, [key, child]) => sum + (key === "cache_control" ? 1 : count(child)),
          0,
        )
      : 0;

test("stabilizes Claude Code's native replay reminders and cache breakpoint", () => {
  const initialSystemMessages = [
    {
      role: "system",
      content: [
        {
          type: "text",
          text: "environment reminder\ndate reminder",
          cache_control: { type: "ephemeral", ttl: "1h" },
        },
      ],
    },
  ];
  const canonical = [
    { role: "user", content: [text("inspect")] },
    { role: "assistant", content: [text("first reply")] },
    { role: "user", content: [text("follow up")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "old result" }] },
    { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t2", content: "current result" }] },
  ];
  const current = [
    { role: "user", content: [text("changed reminder"), text("inspect")] },
    { role: "system", content: "environment reminder" },
    { role: "assistant", content: [text("first reply")] },
    { role: "user", content: [text(account), text("follow up")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "old result" }] },
    { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t2", content: `current result\n${account}` }] },
    {
      role: "system",
      content: [{ type: "text", text: "date reminder", cache_control: { type: "ephemeral", ttl: "1h" } }],
    },
  ];

  const rewritten = rewrite(current, framesFor(canonical), initialSystemMessages);
  assert.equal(rewritten.length, 8);
  assert.equal(((rewritten[1]?.content as Block[])[0]?.text as string), "environment reminder\ndate reminder");
  assert.equal((rewritten[1]?.content as Block[])[0]?.cache_control, undefined);
  assert.deepEqual(
    withoutMarkers(rewritten.filter((message) => message.role !== "system")),
    withoutMarkers(canonical),
  );
  assert.deepEqual((rewritten[5]?.content as Block[]).at(-1)?.cache_control, { type: "ephemeral", ttl: "1h" });
  assert.deepEqual((rewritten.at(-1)?.content as Block[]).at(-1)?.cache_control, { type: "ephemeral", ttl: "1h" });
});

for (const prompt of ["/tmp/pi-clipboard-test.png", "/tmp/pi-clipboard-test.png should we do anything with this?"]) {
  test(`rewrites a string initial user message after a tool round: ${prompt} (issue #31)`, () => {
    const canonical = [
      { role: "user", content: [text(prompt)] },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: { path: "/tmp/pi-clipboard-test.png" } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "image result" }] },
    ];
    const wire: WireMessage[] = structuredClone(canonical);
    wire[0]!.content = prompt;
    assert.deepEqual(rewrite(wire, framesFor(canonical)), canonical);
  });
}

test("preserves canonical content, reminders, and replay cache with a string initial user (issue #31)", () => {
  const marker = { type: "ephemeral", ttl: "1h" };
  const prompt = "/tmp/pi-clipboard-test.png <system-reminder>mine</system-reminder>";
  const canonical = [
    { role: "user", content: [text(prompt)] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: `file says\n${account}` }] },
    { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t2", content: "current result" }] },
  ];
  const initialSystemMessages = [
    { role: "system", content: [{ ...text("environment reminder\ndate reminder"), cache_control: marker }] },
  ];
  const wire: WireMessage[] = structuredClone(canonical);
  wire[0]!.content = `${prompt}\n${memory}`;
  wire[2]!.content = [
    { type: "tool_result", tool_use_id: "t1", content: `file says\n${account}\n${memory}` },
    { ...text(account), cache_control: marker },
  ];
  wire[4]!.content = [
    { type: "tool_result", tool_use_id: "t2", content: `current result\n${memory}${account}` },
  ];
  wire.splice(1, 0, { role: "system", content: [text("changed environment reminder")] });
  wire.push({ role: "system", content: [{ ...text("date reminder"), cache_control: marker }] });

  const rewritten = rewrite(wire, framesFor(canonical), initialSystemMessages);
  assert.deepEqual(withoutMarkers(rewritten), [
    canonical[0],
    { role: "system", content: [text("environment reminder\ndate reminder")] },
    ...canonical.slice(1),
  ]);
  assert.equal((rewritten[0]?.content as Block[])[0]?.cache_control, undefined);
  assert.equal((rewritten[1]?.content as Block[])[0]?.cache_control, undefined);
  assert.deepEqual((rewritten[3]?.content as Block[])[0]?.cache_control, marker);
  assert.deepEqual((rewritten[5]?.content as Block[])[0]?.cache_control, marker);
});

test("rejects missing or malformed initial user messages", () => {
  const frames = framesFor([{ role: "user", content: [text("inspect")] }]);
  const invalid: WireMessage[][] = [
    [],
    [{ role: "assistant", content: "inspect" }],
    [{ role: "system", content: [text("inspect")] }],
    ...[undefined, null, false, true, 0, 42, {}, text("inspect")].map((content) => [{ role: "user", content }]),
  ];
  for (const wire of invalid) {
    assert.throws(() => rewrite(wire, frames), /Claude Code request is missing its initial user message/);
  }
  for (const content of [[], [null], ["inspect"]]) {
    assert.throws(() => rewrite([{ role: "user", content }], frames), /lost a text block in user message 0/);
  }
});

test("keeps the prefix stable when the date reminder sits mid-history", () => {
  const environment = { role: "system", content: "environment reminder\nToday's date is 2026-11-08." };
  const date = { role: "system", content: "Today's date is 2026-11-08." };
  const inspect = { role: "user", content: [text("inspect")] };
  const call = { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] };
  const result = { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "result" }] };
  const reply = { role: "assistant", content: [text("done")] };
  const thanks = { role: "user", content: [text("thanks")] };
  const ok = { role: "assistant", content: [text("ok")] };
  const next = { role: "user", content: [text("next")] };

  // Claude Code appended the date after the tool result, and the process kept it there.
  const first = rewrite(
    [inspect, environment, call, result, date, reply, thanks],
    framesFor([inspect, call, result, reply, thanks]),
    [environment],
  );
  // A later fresh replay of the same history has no date at that position.
  const second = rewrite(
    [inspect, environment, call, result, reply, thanks, ok, next],
    framesFor([inspect, call, result, reply, thanks, ok, next]),
    [environment],
  );

  assert.deepEqual(withoutMarkers(second.slice(0, first.length)), withoutMarkers(first));
});

test("drops a repeated date reminder without captured system messages", () => {
  const canonical = [
    { role: "user", content: [text("inspect")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "result" }] },
    { role: "assistant", content: [text("done")] },
  ];
  const wire = [
    canonical[0]!,
    { role: "system", content: "environment reminder\nToday's date is 2026-11-08." },
    canonical[1]!,
    canonical[2]!,
    { role: "system", content: "Today's date is 2026-11-08." },
    canonical[3]!,
  ];
  const rewritten = rewrite(wire, framesFor(canonical));
  assert.deepEqual(
    rewritten.map((message) => message.role),
    ["user", "system", "assistant", "user", "assistant"],
  );
});

test("moves a new date reminder to the end so warm and fresh turns share a prefix", () => {
  const environment = { role: "system", content: "environment reminder\nToday's date is 2026-11-25." };
  const date = { role: "system", content: "Today's date is 2026-11-26." };
  const first = { role: "user", content: [text("first")] };
  const answer = { role: "assistant", content: [text("answer")] };
  const second = { role: "user", content: [text("second")] };
  const reply = { role: "assistant", content: [text("reply")] };
  const third = { role: "user", content: [text("third")] };
  const history = [first, answer, second, reply, third];

  // The first turn after midnight: Claude Code appends the new date.
  const before = rewrite([first, environment, answer, second, date], framesFor([first, answer, second]), [environment]);
  // A warm process keeps the reminder where that turn ended; a fresh one appends it again.
  const warm = rewrite([first, environment, answer, second, date, reply, third], framesFor(history), [environment]);
  const fresh = rewrite([first, environment, answer, second, reply, third, date], framesFor(history), [environment]);

  assert.deepEqual(warm, fresh);
  assert.deepEqual(withoutMarkers(warm.at(-1)), date);
  assert.deepEqual(withoutMarkers(warm.slice(0, before.length - 1)), withoutMarkers(before.slice(0, -1)));

  // The previous turn cached up to its last user message, which this turn marks too.
  const marked = (messages: WireMessage[]) =>
    messages.flatMap((message, index) =>
      Array.isArray(message.content) && message.content.some((block: Block) => block.cache_control) ? [index] : [],
    );
  assert.deepEqual(marked(before), [3]);
  assert.deepEqual(marked(warm), [3, 5]);
});

test("keeps only the newest date reminder and the rest of its system message", () => {
  const environment = { role: "system", content: "environment reminder\nToday's date is 2026-11-25." };
  const canonical = [
    { role: "user", content: [text("first")] },
    { role: "assistant", content: [text("answer")] },
    { role: "user", content: [text("second")] },
    { role: "assistant", content: [text("reply")] },
    { role: "user", content: [text("third")] },
  ];
  const wire = [
    canonical[0]!,
    environment,
    canonical[1]!,
    canonical[2]!,
    { role: "system", content: [text("Today's date is 2026-11-26."), text("parallel tool nudge")] },
    canonical[3]!,
    canonical[4]!,
    { role: "system", content: "Today's date is 2026-11-27." },
  ];

  assert.deepEqual(withoutMarkers(rewrite(wire, framesFor(canonical), [environment])), [
    canonical[0],
    environment,
    canonical[1],
    canonical[2],
    { role: "system", content: [text("parallel tool nudge")] },
    canonical[3],
    canonical[4],
    { role: "system", content: "Today's date is 2026-11-27." },
  ]);
});

for (const injections of [undefined, [], [{ beforeAssistant: null, content: [text("tool update")] }]]) {
  test(`restores snapshot output_config only for native replay (${JSON.stringify(injections)})`, () => {
    const marker = { type: "ephemeral", ttl: "1h" };
    const snapshot = [
      { role: "system", content: [{ ...text("environment"), cache_control: marker }], output_config: { effort: "medium" } },
      { role: "system", content: "date", output_config: { effort: "high" } },
    ];
    const input = { output_config: "tool argument" };
    const content = [
      { type: "thinking" as const, thinking: "Test reasoning", signature: "Test signature" },
      { type: "redacted_thinking" as const, data: "Test opaque signature" },
      { type: "tool_use" as const, id: "t1", name: "mcp__pi__probe", input },
    ];
    const assistant: AssistantMessage = {
      role: "assistant", api: "test", provider: "claude-code", model: "haiku", timestamp: 0, stopReason: "toolUse",
      content: [{ type: "toolCall", id: "t1", name: "probe", arguments: input }],
      usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    };
    assistant.diagnostics = [makeReplayDiagnostic({ role: "assistant", content }, assistant, snapshot) as never];
    const original = structuredClone(assistant);
    const frames = prepareFrames([
      { role: "user", content: "inspect", timestamp: 0 },
      assistant,
      { role: "toolResult", toolCallId: "t1", toolName: "probe", content: [{ type: "text", text: "Test result" }], isError: false, timestamp: 1 },
    ]);
    const body = {
      output_config: { effort: "low" },
      messages: [
        frames[0]!.message,
        { role: "system", content: [text("new environment")] },
        frames[1]!.message,
        frames[2]!.message,
      ],
    };
    const saved = getInitialSystemMessages([assistant])!;
    const rewritten = JSON.parse(rewriteInitialRequest(
      Buffer.from(JSON.stringify(body)), frames, saved, injections,
    ).toString("utf8"));

    assert.deepEqual(rewritten.output_config, body.output_config);
    const expected = structuredClone(snapshot);
    if (!injections) for (const message of expected) delete (message as Partial<typeof message>).output_config;
    assert.deepEqual(rewritten.messages.slice(1, 3), expected);
    assert.deepEqual(rewritten.messages[3].content, content);
    assert.deepEqual(rewritten.messages.findLast((message: WireMessage) => message.role !== "system").content.at(-1).cache_control, marker);
    assert.deepEqual(saved, snapshot);
    assert.deepEqual(assistant, original);
    if (injections?.length) assert.equal(rewritten.messages.at(-1).content[0].text, "tool update");
  });
}

test("leaves current request output_config alone without a saved snapshot", () => {
  const body = {
    output_config: { effort: "low" },
    messages: [
      { role: "user", content: [text("inspect")] },
      { role: "system", content: "environment", output_config: { effort: "medium" } },
    ],
  };
  assert.deepEqual(
    JSON.parse(rewriteInitialRequest(Buffer.from(JSON.stringify(body)), framesFor(body.messages), undefined).toString("utf8")),
    body,
  );
});

test("strips reminders when the first user message carries none (issue #27)", () => {
  const canonical = [
    { role: "user", content: [text("inspect")] },
    { role: "assistant", content: [text("first reply")] },
    { role: "user", content: [text("follow up")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "result" }] },
  ];
  const wire = [
    { role: "user", content: [text("inspect")] },
    { role: "assistant", content: [text("first reply")] },
    { role: "user", content: [text(memory), text(account), text("follow up")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: `result\n${memory}${account}` }] },
  ];
  // No captured system messages: stripping must not depend on them.
  assert.deepEqual(rewrite(wire, framesFor(canonical)), canonical);
  assert.deepEqual(
    withoutMarkers(
      rewrite(wire, framesFor(canonical), [{ role: "system", content: "environment reminder" }]).filter(
        (message) => message.role !== "system",
      ),
    ),
    canonical,
  );
});

test("strips reminder tails from tool results whatever their separator or subset", () => {
  const canonicalResult = { type: "tool_result", tool_use_id: "t1", content: "result" };
  const frames = framesFor([
    { role: "user", content: [text("inspect")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [canonicalResult] },
  ]);
  for (const tail of [
    `\n${memory}\n${account}`,
    `\n\n${memory}\n\n${account}`,
    `\n\n${account}`,
    `\n${account}`,
    account,
  ]) {
    const rewritten = rewrite(
      [
        { role: "user", content: [text("inspect")] },
        { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: `result${tail}` }] },
      ],
      frames,
    );
    assert.deepEqual((rewritten[2]?.content as Block[])[0]?.content, "result", JSON.stringify(tail));
  }
});

test("strips reminder blocks from text-array tool results and keeps Claude Code's images", () => {
  const image = { type: "image", source: { type: "base64", media_type: "image/png", data: "original" } };
  const resized = { type: "image", source: { type: "base64", media_type: "image/png", data: "resized" } };
  const note = text("[Image: original 4000x4000, displayed at 2000x2000. Multiply coordinates by 2.00 to map to original image.]");
  const frames = framesFor([
    { role: "user", content: [text("look"), image] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: [text("result"), image] }] },
  ]);
  const rewritten = rewrite(
    [
      { role: "user", content: [text(memory), text("look"), resized, note] },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "t1", content: [text("result"), image, text(`\n${account}`)] },
          text(memory),
        ],
      },
    ],
    frames,
  );
  assert.deepEqual(rewritten[0]?.content, [text("look"), resized, note]);
  assert.deepEqual(rewritten[2]?.content, [
    { type: "tool_result", tool_use_id: "t1", content: [text("result"), image] },
  ]);
});

test("keeps Pi's own image scale note exactly once in a tool result", () => {
  const image = { type: "image", source: { type: "base64", media_type: "image/jpeg", data: "resized" } };
  const note = text("[Image: original 2400x1784, displayed at 2000x1487. Multiply coordinates by 1.20 to map to original image.]");
  const result = [text("Script completed\nOutput:\n"), text("[Image saved to /tmp/shot.jpg (image/jpeg, 57.1KB)]"), image, note, text("<console_output>done</console_output>")];
  const canonical = [
    { role: "user", content: [text("inspect")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: result }] },
  ];
  const withReminder = structuredClone(canonical);
  ((withReminder[2]!.content as Block[])[0]!.content as Block[]).push(text(`\n${account}`));
  for (const wire of [structuredClone(canonical), withReminder]) {
    const rewritten = rewrite(wire, framesFor(canonical));
    assert.deepEqual(withoutMarkers(rewritten), canonical);
    const content = (rewritten[2]?.content as Block[])[0]?.content as Block[];
    assert.equal(content.filter((block) => block.text === note.text).length, 1);
  }
});

test("keeps a tool result whose canonical content already has the image scale note", () => {
  const image = { type: "image", source: { type: "base64", media_type: "image/jpeg", data: "shot" } };
  const note = text("[Image: original 728x2490, displayed at 585x2000. Multiply coordinates by 1.24 to map to original image.]");
  const result = { type: "tool_result", tool_use_id: "t1", content: [text("{}"), image, note] };
  const canonical = [
    { role: "user", content: [text("preview")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "html_preview", input: {} }] },
    { role: "user", content: [result] },
  ];
  const rewritten = rewrite(structuredClone(canonical), framesFor(canonical));
  assert.deepEqual(rewritten[2]?.content, [result]);
});

test("keeps user-authored reminder-like text and ordinary output unchanged", () => {
  const canonical = [
    { role: "user", content: [text("<system-reminder>mine</system-reminder>"), text("inspect")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    {
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "t1", content: `file says\n${account}`, is_error: false },
        text("<system-reminder>also mine</system-reminder>"),
      ],
    },
  ];
  const wire = structuredClone(canonical);
  (wire[2]!.content as Block[])[0]!.content = `file says\n${account}\n${memory}`;
  assert.deepEqual(rewrite(wire, framesFor(canonical)), canonical);
  assert.deepEqual(rewrite(structuredClone(canonical), framesFor(canonical)), canonical);
});

test("moves a breakpoint on a dropped reminder to the canonical tail", () => {
  const marker = { type: "ephemeral" };
  const rewritten = rewrite(
    [{ role: "user", content: [text("inspect"), { ...text(memory), cache_control: marker }] }],
    framesFor([{ role: "user", content: [text("inspect")] }]),
  );
  assert.deepEqual(rewritten[0]?.content, [{ ...text("inspect"), cache_control: marker }]);
});

test("refuses requests that do not line up with Pi's history", () => {
  const frames = framesFor([
    { role: "user", content: [text("inspect")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "result" }] },
  ]);
  assert.throws(() => rewrite([{ role: "user", content: [text("inspect")] }], frames), /1 user messages where Pi's history has 2/);
  assert.throws(
    () =>
      rewrite(
        [
          { role: "user", content: [text("inspect")] },
          { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
          { role: "user", content: [{ type: "tool_result", tool_use_id: "other", content: "result" }] },
        ],
        frames,
      ),
    /lost a tool_result block in user message 1/,
  );
});

test("lines up a failed tool result that had images with Claude Code's text-only copy", () => {
  const image = { type: "image" as const, data: "shot", mimeType: "image/png" };
  const frames = prepareFrames([
    { role: "user", content: "inspect", timestamp: 1 },
    {
      role: "assistant",
      api: "test",
      provider: "claude-code",
      model: "sonnet",
      content: [{ type: "toolCall", id: "t1", name: "codemode", arguments: {} }],
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
      stopReason: "toolUse",
      timestamp: 2,
    },
    {
      role: "toolResult",
      toolCallId: "t1",
      toolName: "codemode",
      content: [text("Script failed"), text("[Image saved to /tmp/shot.png]"), image, text("Script error: ENOENT")] as never,
      isError: true,
      timestamp: 3,
    },
  ]);
  const failed = (frames[2]!.message.content as Block[])[0]!;
  // Claude Code drops the images of an error result with array content and
  // joins its text blocks into one; string content passes through unchanged.
  const asSentByClaudeCode = (block: Block): Block => {
    if (block.is_error !== true || !Array.isArray(block.content)) return block;
    const content = block.content as Block[];
    if (content.every((item) => item.type === "text")) return block;
    const texts = content.filter((item) => item.type === "text").map((item) => item.text as string);
    return { ...block, content: texts.length ? [text(texts.join("\n\n"))] : [] };
  };
  const wire = [
    { role: "user", content: [text("inspect")] },
    structuredClone(frames[1]!.message),
    { role: "user", content: [asSentByClaudeCode(structuredClone(failed))] },
  ];
  assert.deepEqual(withoutMarkers(rewrite(wire, frames)[2]?.content), [failed]);
  assert.equal(failed.content, "Script failed\n[Image saved to /tmp/shot.png]\nScript error: ENOENT");
});

test("injects native system history at assistant boundaries", () => {
  const body = {
    system: [
      { type: "text", text: "identity", cache_control: { type: "ephemeral", ttl: "1h" } },
      { type: "text", text: "prompt", cache_control: { type: "ephemeral", ttl: "1h" } },
    ],
    messages: [
      { role: "user", content: [{ type: "text", text: "first" }] },
      { role: "system", content: [{ type: "text", text: "environment" }] },
      { role: "assistant", content: [{ type: "text", text: "answer" }] },
      { role: "user", content: [{ type: "text", text: "second" }] },
      { role: "system", content: [{ type: "text", text: "token reminder" }] },
    ],
  };
  const rewritten = JSON.parse(
    rewriteInitialRequest(Buffer.from(JSON.stringify(body)), framesFor(body.messages), undefined, [
      {
        beforeAssistant: 0,
        content: [{ type: "tool_addition", tool: { type: "tool_reference", name: "mcp__pi__beta" } }],
      },
      {
        beforeAssistant: null,
        content: [{ type: "tool_removal", tool: { type: "tool_reference", name: "mcp__pi__alpha" } }],
      },
    ]).toString("utf8"),
  ) as typeof body;

  assert.deepEqual(rewritten.messages.map((message) => message.role), [
    "user",
    "system",
    "system",
    "assistant",
    "user",
    "system",
    "system",
  ]);
  assert.equal(
    ((rewritten.messages[2]?.content as Array<Record<string, unknown>>)[0] as Record<string, unknown>).type,
    "tool_addition",
  );
  assert.equal(
    ((rewritten.messages.at(-1)?.content as Array<Record<string, unknown>>)[0] as Record<string, unknown>).type,
    "tool_removal",
  );
  assert.equal((rewritten.system[0] as Record<string, unknown>).cache_control, undefined);
  assert.deepEqual((rewritten.system[1] as Record<string, unknown>).cache_control, {
    type: "ephemeral",
    ttl: "1h",
  });
  assert.deepEqual(
    (rewritten.messages[4]?.content as Array<Record<string, unknown>>).at(-1)?.cache_control,
    { type: "ephemeral", ttl: "1h" },
  );
  assert.equal(
    (rewritten.messages.at(-1)?.content as Array<Record<string, unknown>>).at(-1)?.cache_control,
    undefined,
  );
});

test("keeps one breakpoint in a pre-marked multi-block final message (issue #10)", () => {
  // Shape Claude Code sends for Fable after a tool result: a trailing system
  // message whose first block is already marked.
  const marker = { type: "ephemeral", ttl: "1h" };
  const body = {
    system: [
      { type: "text", text: "billing" },
      { type: "text", text: "identity", cache_control: marker },
      { type: "text", text: "prompt", cache_control: marker },
    ],
    tools: [{ name: "mcp__pi__bash", input_schema: { type: "object" }, cache_control: marker }],
    messages: [
      { role: "user", content: [{ type: "text", text: "run it" }] },
      { role: "system", content: "environment" },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "mcp__pi__bash", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "hi" }] },
      {
        role: "system",
        content: [
          { type: "text", text: "date reminder", cache_control: marker },
          { type: "text", text: "parallel tool nudge" },
        ],
      },
    ],
  };
  const rewritten = JSON.parse(
    rewriteInitialRequest(
      Buffer.from(JSON.stringify(body)),
      framesFor(body.messages),
      [{ role: "system", content: [{ type: "text", text: "environment", cache_control: marker }] }],
      [],
    ).toString("utf8"),
  );
  assert.ok(count(rewritten) <= 4, `found ${count(rewritten)} cache_control markers`);
  const trailing = rewritten.messages.at(-1).content as Array<Record<string, unknown>>;
  assert.deepEqual(
    trailing.map((block) => block.cache_control),
    [undefined, undefined],
  );
  assert.deepEqual((rewritten.messages[3].content as Array<Record<string, unknown>>).at(-1)?.cache_control, marker);
});

type WirePayload = { tools?: Block[]; system?: Block[]; messages: WireMessage[] };

function rewritePayload(
  body: WirePayload,
  frames: NativeFrame[],
  initialSystemMessages?: unknown[],
  injections?: Parameters<typeof rewriteInitialRequest>[3],
): WirePayload {
  const output = rewriteInitialRequest(Buffer.from(JSON.stringify(body)), frames, initialSystemMessages, injections);
  return JSON.parse(output.toString("utf8")) as WirePayload;
}

test("keeps four breakpoints when a system update trails the last user message", () => {
  // A Pi system-prompt update recorded just before the newest user message
  // becomes a trailing system message after Claude Code's marked last user.
  const marker = { type: "ephemeral", ttl: "1h" };
  const canonical = [
    { role: "user", content: [text("inspect")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "mcp__pi__read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "one" }] },
    { role: "assistant", content: [{ type: "tool_use", id: "t2", name: "mcp__pi__read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t2", content: "two" }] },
    { role: "assistant", content: [{ type: "tool_use", id: "t3", name: "mcp__pi__read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t3", content: "three" }] },
    { role: "assistant", content: [text("done")] },
    { role: "user", content: [text("follow up")] },
  ];
  const wire = structuredClone(canonical) as WireMessage[];
  wire.splice(1, 0, { role: "system", content: [text("environment")] });
  wire.at(-1)!.content = [text(account), { ...text("follow up"), cache_control: marker }];
  const rewritten = rewritePayload(
    {
      tools: [{ name: "mcp__pi__read", input_schema: { type: "object" }, cache_control: marker }],
      system: [
        { type: "text", text: "identity", cache_control: marker },
        { type: "text", text: "prompt", cache_control: marker },
      ],
      messages: wire,
    },
    framesFor(canonical),
    [{ role: "system", content: [{ type: "text", text: "environment", cache_control: marker }] }],
    [{ beforeAssistant: null, content: [text("system prompt section updated")] }],
  );

  assert.ok(count(rewritten) <= 4, `found ${count(rewritten)} cache_control markers`);
  assert.equal(rewritten.messages.at(-1)?.role, "system");
  assert.equal(count(rewritten.messages.at(-1)), 0);
  const users = rewritten.messages.filter((message) => message.role === "user");
  assert.deepEqual((users.at(-1)?.content as Block[]).at(-1)?.cache_control, marker);
  assert.deepEqual(
    withoutMarkers(rewritten.messages.filter((message) => message.role !== "system")),
    canonical,
  );
});

test("never emits more than four breakpoints across replay shapes", () => {
  const marker = { type: "ephemeral", ttl: "1h" };
  let checked = 0;
  for (const userCount of [1, 2, 3, 5]) {
    for (const nativeMarker of ["none", "last user", "trailing system", "both"]) {
      for (const trailingInjection of [false, true]) {
        for (const assistantInjection of [false, true]) {
          for (const withInitial of [false, true]) {
            for (const withInjections of [false, true]) {
              for (const toolMarker of [false, true]) {
                for (const systemMarkers of [1, 2]) {
                  // Not constructible: no assistant to inject before, or
                  // injections without an injections argument.
                  if (assistantInjection && userCount < 2) continue;
                  if (!withInjections && (trailingInjection || assistantInjection)) continue;

                  const canonical: WireMessage[] = [{ role: "user", content: [text("inspect")] }];
                  for (let turn = 1; turn < userCount; turn++) {
                    const id = `t${turn}`;
                    canonical.push(
                      { role: "assistant", content: [{ type: "tool_use", id, name: "mcp__pi__read", input: {} }] },
                      { role: "user", content: [{ type: "tool_result", tool_use_id: id, content: `result ${turn}` }] },
                    );
                  }
                  const wire = structuredClone(canonical);
                  wire.splice(1, 0, { role: "system", content: [text("environment")] });
                  if (nativeMarker === "last user" || nativeMarker === "both") {
                    const last = wire.findLast((message) => message.role === "user")!.content as Block[];
                    last[last.length - 1] = { ...last.at(-1)!, cache_control: marker };
                  }
                  if (nativeMarker === "trailing system" || nativeMarker === "both") {
                    wire.push({ role: "system", content: [{ ...text("token reminder"), cache_control: marker }] });
                  }
                  const injections = withInjections
                    ? [
                        ...(assistantInjection ? [{ beforeAssistant: 0, content: [text("tools changed")] }] : []),
                        ...(trailingInjection ? [{ beforeAssistant: null, content: [text("prompt updated")] }] : []),
                      ]
                    : undefined;
                  const system = [
                    { type: "text", text: "identity", ...(systemMarkers === 2 ? { cache_control: marker } : {}) },
                    { type: "text", text: "prompt", cache_control: marker },
                  ];
                  const shape = JSON.stringify({
                    userCount,
                    nativeMarker,
                    trailingInjection,
                    assistantInjection,
                    withInitial,
                    withInjections,
                    toolMarker,
                    systemMarkers,
                  });

                  const rewritten = rewritePayload(
                    {
                      tools: [
                        { name: "mcp__pi__read", input_schema: { type: "object" }, ...(toolMarker ? { cache_control: marker } : {}) },
                      ],
                      system,
                      messages: wire,
                    },
                    framesFor(canonical),
                    withInitial ? [{ role: "system", content: [{ ...text("environment"), cache_control: marker }] }] : undefined,
                    injections,
                  );
                  assert.ok(count(rewritten) <= 4, `${shape}: found ${count(rewritten)} cache_control markers`);
                  if (withInitial || withInjections) {
                    const final = rewritten.messages.findLast((message) => message.role !== "system");
                    assert.deepEqual((final?.content as Block[]).at(-1)?.cache_control, marker, shape);
                  }
                  assert.deepEqual(
                    withoutMarkers(rewritten.messages.filter((message) => message.role !== "system")),
                    canonical,
                    shape,
                  );
                  checked += 1;
                }
              }
            }
          }
        }
      }
    }
  }
  assert.equal(checked, 576, "every constructible shape is checked");
});

test("keeps four breakpoints on warm and cold continuations (issue #40 repro)", () => {
  // Request shape captured from Claude Code 2.1.330, content replaced. Claude
  // Code marks only its last message: the newest user on a warm worker, a
  // trailing date reminder on a cold one.
  const marker = { type: "ephemeral", ttl: "1h" };
  const marked = (value: string): Block => ({ ...text(value), cache_control: marker });
  const tools = [
    { name: "mcp__pi__read", input_schema: { type: "object" }, cache_control: marker },
    { name: "__pi_deferred_placeholder__", input_schema: { type: "object" }, defer_loading: true },
  ];
  const system = [text("billing header"), marked("identity"), marked("system prompt")];
  const initialSystemMessages = [{ role: "system", content: [marked("environment reminder\ntoday's date")] }];
  const frames = (users: number): NativeFrame[] => {
    const out: NativeFrame[] = [];
    for (let i = 1; i <= users; i++) {
      out.push({ type: "user", message: { role: "user", content: [text(`question ${i}`)] } });
      if (i < users) out.push({ type: "assistant", message: { role: "assistant", content: [text(`answer ${i}`)] } });
    }
    return out;
  };
  const update = [{ beforeAssistant: null, content: [text('Updated system prompt section "mcp_servers": ...')] }];

  for (const users of [2, 4]) {
    for (const warm of [true, false]) {
      for (const withUpdate of [false, true]) {
        const shape = `${users} users, ${warm ? "warm" : "cold"}, ${withUpdate ? "with" : "without"} trailing update`;
        const messages: WireMessage[] = [];
        frames(users).forEach((frame, index) => {
          messages.push(structuredClone(frame.message) as WireMessage);
          if (index === 0) messages.push({ role: "system", content: "environment reminder" });
        });
        if (warm) messages.at(-1)!.content = [marked(`question ${users}`)];
        else messages.push({ role: "system", content: [marked("today's date")] });
        const body = { tools: structuredClone(tools), system: structuredClone(system), messages };
        assert.equal(count(body), 4, shape);

        const rewritten = rewritePayload(body, frames(users), initialSystemMessages, withUpdate ? update : []);
        assert.ok(count(rewritten) <= 4, `${shape}: found ${count(rewritten)} cache_control markers`);
      }
    }
  }
});

const userFrame = (text: string) => ({
  type: "user" as const,
  message: { role: "user" as const, content: [{ type: "text", text }] },
});

for (const key of [
  "ANTHROPIC_AUTH_TOKEN",
  "ANTHROPIC_BASE_URL",
  "ANTHROPIC_FOUNDRY_API_KEY",
  "CLAUDE_CODE_USE_BEDROCK",
  "CLAUDE_CODE_USE_VERTEX",
  "CLAUDE_CODE_USE_FOUNDRY",
]) {
  test(`rejects ${key} for model discovery and requests`, async () => {
    const env = { [key]: "1", ANTHROPIC_API_KEY: "pi-test-dummy" };
    const error = new RegExp(`refuses conflicting auth/backend overrides:.*${key}`);
    await assert.rejects(discoverClaudeModels(env), error);
    await assert.rejects(
      runClaudeRequest({
        model: "sonnet",
        frames: [],
        systemPrompt: "",
        manifest: [],
        extraBody: {},
        env,
      }),
      error,
    );
  });
}

test("runs a request through a stream-json Claude executable with an inherited API key", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-claude-code-test-"));
  const workingRoot = await realpath(root);
  const script = join(root, "fake-claude.mjs");
  await writeFile(
    script,
    `#!/usr/bin/env node
import readline from "node:readline";
import {appendFileSync, readFileSync} from "node:fs";
if ("ANTHROPIC_API_KEY" in process.env) throw new Error("API key leaked to Claude child");
if (process.argv.includes("auth") && process.argv.includes("status")) {
  console.log(JSON.stringify({loggedIn: true, authMethod: "claude.ai", subscriptionType: "test"}));
  process.exit(0);
}
if (process.argv.includes("--system-prompt-file")) {
  for (const name of ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "DISABLE_TELEMETRY", "DISABLE_ERROR_REPORTING"]) {
    if (process.env[name] !== "1") throw new Error(\`worker privacy flag missing: \${name}\`);
  }
}
if (process.cwd() !== process.env.EXPECTED_CWD) throw new Error(\`wrong cwd: \${process.cwd()}\`);
if (process.env.CLAUDE_CODE_ENTRYPOINT !== "pi-agent") throw new Error(\`wrong entrypoint: \${process.env.CLAUDE_CODE_ENTRYPOINT}\`);
if (process.env.CLAUDE_CODE_DISABLE_FILE_MENTIONS !== "1") throw new Error("file mentions enabled");
if (process.env.CLAUDE_CODE_SILENT_TURN_REMINDER !== "off") throw new Error("silent turn reminder enabled");
if (process.env.PROCESS_LOG) appendFileSync(process.env.PROCESS_LOG, \`\${process.pid}\\n\`);
const systemIndex = process.argv.indexOf("--system-prompt-file");
if (systemIndex >= 0 && readFileSync(process.argv[systemIndex + 1], "utf8") !== process.env.EXPECTED_SYSTEM_PROMPT) {
  throw new Error("wrong system prompt");
}
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
let queryCount = 0;
// Real Claude Code echoes a frame's uuid on that frame's result.
const echoUuid = frame => frame.uuid ? {user_message_uuids: [frame.uuid]} : {};
const strayResult = {type: "result", subtype: "error_during_execution", is_error: true, num_turns: 0};
for await (const line of input) {
  const frame = JSON.parse(line);
  if (frame.type === "control_request" && frame.request?.subtype === "initialize") {
    console.log(JSON.stringify({
      type: "control_response",
      response: {
        subtype: "success",
        request_id: frame.request_id,
        response: {
          models: [
            {value: "default", resolvedModel: "claude-opus-5-5[1m]", displayName: "Default", description: "Opus with 1M context"},
            {value: "sonnet", resolvedModel: "claude-sonnet-5", displayName: "Sonnet", description: "Efficient", supportsAdaptiveThinking: true}
          ]
        }
      }
    }));
    continue;
  }
  if (frame.shouldQuery === false) {
    console.log(JSON.stringify({type: "result", num_turns: 0, ...echoUuid(frame)}));
    continue;
  }
  if (frame.type !== "user") continue;
  queryCount += 1;
  const requestedTool = frame.message?.content?.some?.(block => block.type === "text" && block.text === "call-tool");
  const text = queryCount === 1 ? "ok" : \`ok\${queryCount}\`;
  const id = queryCount === 1 ? "m_fake" : \`m_fake_\${queryCount}\`;
  const usage = {input_tokens: 3, output_tokens: 1, cache_read_input_tokens: 0, cache_creation_input_tokens: 0};
  if (requestedTool) {
    const content = [{type: "tool_use", id: "tool_fake", name: "mcp__pi__echo", input: {value: "test"}}];
    console.log(JSON.stringify({type: "stream_event", event: {type: "message_stop"}}));
    console.log(JSON.stringify({type: "assistant", message: {id, role: "assistant", content, stop_reason: "tool_use", usage}}));
    console.log(JSON.stringify({type: "result", subtype: "error_max_turns", is_error: true, usage, ...echoUuid(frame)}));
    continue;
  }
  if (process.env.NATIVE_ERROR) {
    const content = [{type: "text", text: "Usage limit"}];
    console.log(JSON.stringify({type: "assistant", error: "rate_limit", message: {id, role: "assistant", content, usage}}));
    console.log(JSON.stringify({type: "result", subtype: "success", is_error: true, usage, ...echoUuid(frame)}));
    continue;
  }
  const events = [
    {type: "message_start", message: {id, role: "assistant", content: [], usage}},
    {type: "content_block_start", index: 0, content_block: {type: "text", text: ""}},
    {type: "content_block_delta", index: 0, delta: {type: "text_delta", text}},
    {type: "content_block_stop", index: 0},
    {type: "message_delta", delta: {stop_reason: "end_turn"}, usage},
    {type: "message_stop"}
  ];
  for (const event of events) console.log(JSON.stringify({type: "stream_event", event}));
  console.log(JSON.stringify({type: "assistant", message: {id, role: "assistant", content: [{type: "text", text}], stop_reason: "end_turn", usage}}));
  console.log(JSON.stringify({type: "result", subtype: "success", is_error: false, usage, ...echoUuid(frame)}));
  // A result no turn waits for stays buffered until the next turn reads it.
  if (process.env.STALE_RESULT === "foreign") console.log(JSON.stringify({...strayResult, user_message_uuids: ["other"]}));
  if (process.env.STALE_RESULT === "bare") console.log(JSON.stringify(strayResult));
}
`,
    "utf8",
  );
  // Windows cannot execute a shebang script directly; go through a .cmd shim,
  // which also exercises the extension's cmd.exe launch path.
  const executable = process.platform === "win32" ? join(root, "fake-claude.cmd") : script;
  if (executable !== script) {
    await writeFile(executable, `@ECHO off\r\n"${process.execPath}" "${script}" %*\r\n`, "utf8");
  }
  await chmod(executable, 0o755);

  const events: string[] = [];
  const originalApiKey = process.env.ANTHROPIC_API_KEY;
  const originalAgentDir = process.env.PI_CODING_AGENT_DIR;
  try {
    process.env.ANTHROPIC_API_KEY = "pi-test-dummy";
    // Keep the saved login status out of the real agent directory.
    process.env.PI_CODING_AGENT_DIR = join(root, "agent");
    const provider = createClaudeCodeProvider();
    const authCheck = await provider.auth.apiKey?.check?.({
      ctx: {
        env: async (name) => (name === "PI_CLAUDE_CODE_COMMAND" ? executable : undefined),
        fileExists: async () => false,
      },
      signal: new AbortController().signal,
    });
    assert.equal(authCheck?.type, "oauth");
    assert.equal(provider.auth.oauth?.isSubscription, true);
    await assert.rejects(
      provider.auth.oauth!.login({
        signal: new AbortController().signal,
        prompt: async () => "",
        notify: () => {},
      }),
      /claude auth login/,
    );

    const explicitEnv = Object.freeze({
      PATH: process.env.PATH,
      PI_CLAUDE_CODE_COMMAND: executable,
      EXPECTED_CWD: process.cwd(),
      ANTHROPIC_API_KEY: "pi-test-explicit-dummy",
    });
    assert.equal((await getClaudeStatus(explicitEnv)).subscription, true);
    for (const env of [
      { PI_CLAUDE_CODE_COMMAND: executable, EXPECTED_CWD: process.cwd() },
      explicitEnv,
    ]) {
      const discovered = await discoverClaudeModels(env, undefined, 5_000);
      assert.deepEqual(discovered.map((model) => model.value), ["default", "sonnet"]);
    }

    const result = await runClaudeRequest({
      model: "sonnet",
      cwd: workingRoot,
      frames: [
        { type: "user", message: { role: "user", content: [{ type: "text", text: "old" }] }, shouldQuery: false },
        { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "old reply" }] } },
        { type: "user", message: { role: "user", content: [{ type: "text", text: "new" }] } },
      ],
      systemPrompt: "system",
      manifest: [],
      extraBody: { tools: [] },
      env: {
        PI_CLAUDE_CODE_COMMAND: executable,
        EXPECTED_CWD: workingRoot,
        EXPECTED_SYSTEM_PROMPT: "system",
      },
      timeoutMs: 5_000,
      onEvent: (event) => events.push(event.type),
    });
    assert.equal(result.message.id, "m_fake");
    assert.equal(result.message.stop_reason, "end_turn");
    assert.equal(result.usage.input_tokens, 3);
    assert.equal(result.relay.upstreamRequests, 0);
    assert.equal(events.at(-1), "message_stop");

    const processLog = join(root, "processes.log");
    const common = {
      model: "sonnet",
      cwd: workingRoot,
      systemPrompt: "system",
      manifest: [],
      extraBody: { tools: [] },
      env: {
        PI_CLAUDE_CODE_COMMAND: executable,
        EXPECTED_CWD: workingRoot,
        EXPECTED_SYSTEM_PROMPT: "system",
        PROCESS_LOG: processLog,
        ANTHROPIC_API_KEY: explicitEnv.ANTHROPIC_API_KEY,
      },
      timeoutMs: 5_000,
    };
    const first = await runClaudeRequest({
      ...common,
      sessionId: "stable-session",
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "one" }] } }],
    });
    const second = await runClaudeRequest({
      ...common,
      sessionId: "stable-session",
      frames: [
        {
          type: "user",
          message: { role: "user", content: [{ type: "text", text: "one" }] },
          shouldQuery: false,
        },
        // prepareFrames tags replayed turns with their model; that must not break continuation.
        { type: "assistant", message: { role: "assistant", content: first.message.content, model: "opus" } },
        { type: "user", message: { role: "user", content: [{ type: "text", text: "two" }] } },
      ],
    });
    assert.equal(second.message.id, "m_fake_2");
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 1);
    assert.equal(cachedClaudeProcessCount(), 1);

    // A branch mismatch replaces the warm process instead of risking stale
    // hidden Claude state.
    const branched = await runClaudeRequest({
      ...common,
      sessionId: "stable-session",
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "branch" }] } }],
    });
    assert.equal(branched.message.id, "m_fake");
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 2);

    // Retained workers are deliberately bounded. Creating two more sessions
    // evicts the least-recently-used idle worker and leaves only two alive.
    for (const sessionId of ["second-session", "third-session"]) {
      await runClaudeRequest({
        ...common,
        sessionId,
        frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: sessionId }] } }],
      });
    }
    assert.equal(cachedClaudeProcessCount(), 2);
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 4);

    // A tool call makes Claude's hidden history diverge because its inert MCP
    // server has already returned a denial. Never retain that process; replay
    // the real Pi tool result in a fresh process on the next request.
    const toolCommon = {
      ...common,
      manifest: [
        {
          name: "echo",
          description: "Echo a value",
          inputSchema: { type: "object", properties: { value: { type: "string" } } },
        },
      ],
      extraBody: {
        tools: [
          {
            name: "mcp__pi__echo",
            description: "Echo a value",
            input_schema: { type: "object", properties: { value: { type: "string" } } },
          },
        ],
      },
    };
    const toolCall = await runClaudeRequest({
      ...toolCommon,
      sessionId: "tool-session",
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "call-tool" }] } }],
    });
    assert.equal(toolCall.message.stop_reason, "tool_use");
    assert.equal(cachedClaudeProcessCount(), 1);
    const afterTool = await runClaudeRequest({
      ...toolCommon,
      sessionId: "tool-session",
      frames: [
        {
          type: "user",
          message: { role: "user", content: [{ type: "text", text: "call-tool" }] },
          shouldQuery: false,
        },
        { type: "assistant", message: { role: "assistant", content: toolCall.message.content } },
        {
          type: "user",
          message: {
            role: "user",
            content: [{ type: "tool_result", tool_use_id: "tool_fake", content: "echoed" }],
          },
        },
      ],
    });
    assert.equal(afterTool.message.id, "m_fake");
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 6);
    assert.equal(explicitEnv.ANTHROPIC_API_KEY === "pi-test-explicit-dummy", true);
    assert.equal(common.env.ANTHROPIC_API_KEY === "pi-test-explicit-dummy", true);
    assert.equal(process.env.ANTHROPIC_API_KEY === "pi-test-dummy", true);

    // A late result for another user message must not end the next warm turn.
    const warmTurns = async (sessionId: string, extra: Record<string, string>) => {
      const warm = { ...common, env: { ...common.env, ...extra }, sessionId };
      const one = await runClaudeRequest({ ...warm, frames: [userFrame("one")] });
      return runClaudeRequest({
        ...warm,
        frames: [
          { ...userFrame("one"), shouldQuery: false },
          { type: "assistant", message: { role: "assistant", content: one.message.content } },
          userFrame("two"),
        ],
      });
    };
    const foreign = await warmTurns("foreign-session", { STALE_RESULT: "foreign" });
    assert.equal(foreign.message.id, "m_fake_2");
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 7);

    // A stray result without a uuid fails the warm turn before any output, so
    // it is replayed once on a fresh process.
    const bare = await warmTurns("bare-session", { STALE_RESULT: "bare" });
    assert.equal(bare.message.id, "m_fake");
    assert.equal((await readFile(processLog, "utf8")).trim().split("\n").length, 9);

    // A native error assistant reports its text, not an incomplete response.
    await assert.rejects(
      runClaudeRequest({ ...common, env: { ...common.env, NATIVE_ERROR: "1" }, frames: [userFrame("one")] }),
      /Claude Code API error: Usage limit/,
    );
  } finally {
    if (originalApiKey === undefined) delete process.env.ANTHROPIC_API_KEY;
    else process.env.ANTHROPIC_API_KEY = originalApiKey;
    if (originalAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = originalAgentDir;
    await closeClaudeProcesses();
    await rm(root, { recursive: true, force: true });
  }
});

/**
 * Run one turn through a fake Claude Code that, like the real one, reads the
 * relayed response and retries once natively. `upstream` answers the relayed
 * request; without it the relay points at a closed port.
 */
async function failingUpstreamTurn(
  upstream?: (request: http.IncomingMessage, response: http.ServerResponse) => void,
): Promise<{ message: string; retryable: boolean }> {
  let server: http.Server | undefined;
  let port: number;
  if (upstream) {
    server = http.createServer(upstream);
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("missing address");
    port = address.port;
  } else {
    // Reserve a port, then close it so connecting is refused.
    const probe = http.createServer();
    await new Promise<void>((resolve) => probe.listen(0, "127.0.0.1", resolve));
    port = (probe.address() as { port: number }).port;
    await new Promise<void>((resolve) => probe.close(() => resolve()));
  }

  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-claude-code-upstream-")));
  const executable = join(root, "fake-claude.mjs");
  await writeFile(
    executable,
    `#!/usr/bin/env node
import readline from "node:readline";
if (process.argv.includes("auth") && process.argv.includes("status")) {
  console.log(JSON.stringify({loggedIn: true, authMethod: "claude.ai", subscriptionType: "max"}));
  process.exit(0);
}
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
for await (const line of input) {
  const frame = JSON.parse(line);
  if (frame.type !== "user") continue;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await fetch(\`\${process.env.ANTHROPIC_BASE_URL}/v1/messages\`, {
        method: "POST",
        body: JSON.stringify({messages: [frame.message]}),
      });
      await response.text();
    } catch {}
  }
  console.log(JSON.stringify({type: "result", subtype: "error_during_execution", is_error: true}));
}
`,
    "utf8",
  );
  await chmod(executable, 0o755);

  try {
    const error = await runClaudeRequest({
      model: "sonnet",
      cwd: root,
      frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "hi" }] } }],
      systemPrompt: "system",
      manifest: [],
      extraBody: { tools: [] },
      env: { PI_CLAUDE_CODE_COMMAND: executable },
      upstreamUrl: `http://127.0.0.1:${port}`,
      timeoutMs: 5_000,
    }).then(
      () => assert.fail("expected the upstream failure to fail the turn"),
      (error: Error) => error,
    );
    const failed = { role: "assistant", stopReason: "error", errorMessage: error.message } as AssistantMessage;
    return { message: error.message, retryable: isRetryableAssistantError(failed) };
  } finally {
    closeClaudeProcesses();
    if (server) await new Promise<void>((resolve) => server!.close(() => resolve()));
    await rm(root, { recursive: true, force: true });
  }
}

test("reports a dropped upstream stream as a retryable error", async () => {
  // Anthropic answers 200, starts streaming, then the connection drops.
  const { message, retryable } = await failingUpstreamTurn(async (request, response) => {
    for await (const _chunk of request);
    response.writeHead(200, { "content-type": "text/event-stream" });
    const usage = { input_tokens: 2, output_tokens: 0 };
    response.write(
      `event: message_start\ndata: ${JSON.stringify({ type: "message_start", message: { id: "m1", role: "assistant", content: [], usage } })}\n\n`,
    );
    setTimeout(() => response.socket?.destroy(), 20);
  });
  assert.match(message, /status 200, capture incomplete/);
  assert.match(message, /native retries denied 1/);
  assert.equal(retryable, true);
});

test("reports an upstream connection lost before a response as a retryable error", async () => {
  // The connection drops before Anthropic sends a status.
  const hangUp = await failingUpstreamTurn(async (request) => {
    for await (const _chunk of request);
    request.socket.destroy();
  });
  assert.match(hangUp.message, /connection lost before a response \(status unknown/);
  assert.match(hangUp.message, /relay failure Error: socket hang up/);
  assert.equal(hangUp.retryable, true);

  // The connection is refused outright; the reason lives in the error code.
  const refused = await failingUpstreamTurn();
  assert.match(refused.message, /connection lost before a response \(status unknown/);
  assert.match(refused.message, /ECONNREFUSED/);
  assert.equal(refused.retryable, true);
});

test("reports upstream 5xx as retryable and keeps 4xx non-retryable", async () => {
  // 529 (overloaded) is not in Pi's status list and may come without a body.
  const overloaded = await failingUpstreamTurn(async (request, response) => {
    for await (const _chunk of request);
    response.writeHead(529).end();
  });
  assert.match(overloaded.message, /upstream server error \(status 529/);
  assert.equal(overloaded.retryable, true);

  const invalid = await failingUpstreamTurn(async (request, response) => {
    for await (const _chunk of request);
    response
      .writeHead(400, { "content-type": "application/json" })
      .end(JSON.stringify({ type: "error", error: { type: "invalid_request_error", message: "bad request" } }));
  });
  assert.match(invalid.message, /Incomplete Claude Code upstream response \(status 400.*upstream said: bad request/);
  assert.equal(invalid.retryable, false);
});

test("drops Claude Code's date reminder from the middle without captured system messages", () => {
  const date = (day: string) => ({ role: "system", content: `Today's date is 2026-11-${day}.` });
  const history = [
    { role: "user", content: [text("first")] },
    { role: "system", content: [text("environment")] },
    { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "mcp__pi__read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "file" }] },
  ];
  const answer = { role: "assistant", content: [text("done")] };
  const next = { role: "user", content: [text("next")] };
  // A warm process keeps the reminder where its previous request ended; a fresh one appends it.
  const warm = rewrite([...history, date("25"), answer, next], framesFor([...history, answer, next]));
  const fresh = rewrite([...history, answer, next, date("25")], framesFor([...history, answer, next]));

  assert.deepEqual(withoutMarkers(warm), withoutMarkers(fresh));
  assert.deepEqual(withoutMarkers(fresh.at(-1)), date("25"));
  assert.deepEqual(withoutMarkers(warm[1]), history[1]);
});
