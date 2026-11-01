import assert from "node:assert/strict";
import { once } from "node:events";
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { closeClaudeProcesses, runClaudeRequest } from "../src/claude-cli.ts";

test("replays a tool boundary through the relay when Claude's first user content is a string", async (t) => {
  const cache = { type: "ephemeral", ttl: "1h" };
  const canonical = [{ type: "text", text: "Read fixture.txt and summarize it." }];
  const reminder = "<system-reminder>Use the supplied tool output.</system-reminder>";
  const prefix = [{ type: "text", text: reminder }];
  const date = "Fixture date: 2030-01-02";
  const initialSystemMessages = [
    {
      role: "system",
      content: [{ type: "text", text: `Fixture environment\n${date}`, cache_control: cache }],
    },
  ];
  const system = [
    { type: "text", text: "Fixture identity", cache_control: cache },
    { type: "text", text: "Fixture instructions", cache_control: cache },
  ];
  const toolCall = {
    type: "tool_use", id: "toolu_offline_original", name: "mcp__pi__read", input: { path: "fixture.txt" },
  };
  const toolResult = { type: "tool_result", tool_use_id: toolCall.id, content: "Synthetic fixture contents." };
  const usage = { input_tokens: 11, output_tokens: 4, cache_read_input_tokens: 5, cache_creation_input_tokens: 0 };
  const events = [
    {
      type: "message_start",
      message: {
        id: "msg_offline_replay", type: "message", role: "assistant", model: "claude-offline-fixture",
        content: [], stop_reason: null, stop_sequence: null, usage: { ...usage, output_tokens: 0 },
      },
    },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Fixture replay succeeded." } },
    { type: "content_block_stop", index: 0 },
    { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 4 } },
    { type: "message_stop" },
  ];
  const requests: Array<{ method?: string; url?: string; body: string }> = [];
  const upstream = createServer(async (request, response) => {
    try {
      request.setEncoding("utf8");
      let body = "";
      for await (const chunk of request) body += chunk;
      requests.push({ method: request.method, url: request.url, body });
      response.writeHead(200, { "content-type": "text/event-stream", "request-id": "req_offline_replay" });
      response.end(events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""));
    } catch {
      response.destroy();
    }
  });
  const root = await mkdtemp(join(tmpdir(), "pi-claude-replay-"));
  const sessionId = `offline-replay:${root}`;
  t.after(async () => {
    try {
      await closeClaudeProcesses(sessionId);
    } finally {
      upstream.closeAllConnections();
      await Promise.all([
        new Promise<void>((resolve) => upstream.close(() => resolve())),
        rm(root, { recursive: true, force: true }),
      ]);
    }
  });
  upstream.listen(0, "127.0.0.1");
  await once(upstream, "listening");
  const address = upstream.address();
  assert.ok(address && typeof address !== "string");

  const executable = join(root, "fake-claude.mjs");
  await writeFile(executable, `#!/usr/bin/env node
import assert from "node:assert/strict";
import { createInterface } from "node:readline";
const emit = (value) => console.log(JSON.stringify(value));
if (process.argv.includes("auth") && process.argv.includes("status")) {
  emit({loggedIn: true, authMethod: "claude.ai", subscriptionType: "test"});
  process.exit(0);
}
const baseUrl = new URL(process.env.ANTHROPIC_BASE_URL);
assert.equal(baseUrl.protocol, "http:");
assert.equal(baseUrl.hostname, "127.0.0.1");
const history = [];
for await (const line of createInterface({input: process.stdin, crlfDelay: Infinity})) {
  const frame = JSON.parse(line);
  history.push(frame.message);
  if (frame.shouldQuery === false) {
    assert.equal(history.length, 1);
    emit({type: "result", subtype: "success", is_error: false, num_turns: 0});
    continue;
  }
  if (frame.type !== "user") continue;
  assert.deepEqual(history.map(message => message.role), ["user", "assistant", "user"]);
  const result = structuredClone(frame.message);
  result.content = [...${JSON.stringify(prefix)}, ...result.content];
  result.content.at(-1).content += "\\n" + ${JSON.stringify(reminder)};
  const body = {
    model: "claude-offline-fixture", stream: true, max_tokens: 64,
    system: ${JSON.stringify(system)},
    messages: [
      // Claude Code 2.1.316 serializes the replayed initial user as a STRING.
      {...history[0], content: history[0].content.map(block => block.text).join("\\n")},
      {role: "system", content: "Changed environment reminder"},
      history[1],
      result,
      {role: "system", content: [{type: "text", text: ${JSON.stringify(date)}, cache_control: ${JSON.stringify(cache)}}]}
    ]
  };
  const response = await fetch(baseUrl.href + "/v1/messages", {
    method: "POST",
    headers: {"content-type": "application/json", "anthropic-version": "2023-06-01"},
    body: JSON.stringify(body)
  });
  await response.text();
  // No assistant/stream_event shortcut: runClaudeRequest must capture the relay SSE.
  emit({type: "result", subtype: response.ok ? "success" : "error_during_execution",
    is_error: !response.ok, num_turns: 1, usage: ${JSON.stringify(usage)}});
}
`, "utf8");
  await chmod(executable, 0o755);

  const observed: string[] = [];
  const output = await runClaudeRequest({
    sessionId,
    model: "sonnet",
    cwd: root,
    frames: [
      { type: "user", message: { role: "user", content: canonical }, shouldQuery: false },
      { type: "assistant", message: { role: "assistant", content: [toolCall] } },
      { type: "user", message: { role: "user", content: [toolResult] } },
    ],
    systemPrompt: "Offline replay fixture",
    manifest: [{ name: "read", description: "Read a fixture", inputSchema: { type: "object" } }],
    extraBody: { tools: [] },
    toolNames: new Set(["read"]),
    initialUserPrefix: prefix,
    initialSystemMessages,
    upstreamUrl: `http://127.0.0.1:${address.port}`,
    env: { PI_CLAUDE_CODE_COMMAND: executable, PI_CLAUDE_CODE_CONFIG_DIR: root },
    timeoutMs: 5_000,
    onEvent: (event) => observed.push(event.type),
  });

  assert.equal(requests.length, 1);
  assert.equal(requests[0]!.method, "POST");
  assert.equal(requests[0]!.url, "/v1/messages");
  // Full equality checks the original tool ID/result, stable reminders, and all four cache markers.
  assert.deepEqual(JSON.parse(requests[0]!.body), {
    model: "claude-offline-fixture", stream: true, max_tokens: 64, system,
    messages: [
      { role: "user", content: [...prefix, ...canonical] },
      ...initialSystemMessages,
      { role: "assistant", content: [toolCall] },
      { role: "user", content: [{ ...toolResult, cache_control: cache }] },
    ],
  });
  assert.deepEqual(output.relay, { upstreamRequests: 1, blockedRequests: 0, requestId: "req_offline_replay" });
  assert.deepEqual(output.message.content, [{ type: "text", text: "Fixture replay succeeded." }]);
  assert.equal(output.message.id, "msg_offline_replay");
  assert.equal(output.message.stop_reason, "end_turn");
  assert.deepEqual(output.usage, usage);
  assert.equal(output.result.subtype, "success");
  assert.equal(output.result.is_error, false);
  assert.deepEqual(output.initialUserPrefix, prefix);
  assert.deepEqual(output.initialSystemMessages, initialSystemMessages);
  assert.deepEqual(observed, events.map((event) => event.type));
});
