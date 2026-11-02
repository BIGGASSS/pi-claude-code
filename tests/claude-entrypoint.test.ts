import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { discoverClaudeModels, getClaudeStatus, runClaudeRequest } from "../src/claude-cli.ts";
import { claudeCodeEnv, createClaudeCodeProvider } from "../src/provider.ts";

test("identifies Claude subprocesses without inheriting the host's entrypoint", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-claude-entrypoint-"));
  const executable = join(root, "fake-claude.mjs");
  const logPath = join(root, "entrypoints.jsonl");
  await writeFile(
    executable,
    `#!/usr/bin/env node
import { appendFileSync } from "node:fs";
import readline from "node:readline";
const emit = value => console.log(JSON.stringify(value));
const kind = process.argv.includes("auth") ? "auth" : process.argv.includes("--settings") ? "turn" : "discovery";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ kind, entrypoint: process.env.CLAUDE_CODE_ENTRYPOINT }) + "\\n");
if (kind === "auth") {
  emit({ loggedIn: true, authMethod: "claude.ai", subscriptionType: "test" });
  process.exit(0);
}
for await (const line of readline.createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  const frame = JSON.parse(line);
  if (frame.type === "control_request") {
    emit({ type: "control_response", response: {
      subtype: "success", request_id: frame.request_id,
      response: { models: [{ value: "sonnet", displayName: "Sonnet", description: "Test" }] }
    } });
  } else if (frame.type === "user" && frame.shouldQuery !== false) {
    const usage = { input_tokens: 1, output_tokens: 1 };
    emit({ type: "stream_event", event: { type: "message_stop" } });
    emit({ type: "assistant", message: {
      id: "entrypoint-test", role: "assistant", content: [{ type: "text", text: "ok" }], stop_reason: "end_turn", usage
    } });
    emit({ type: "result", subtype: "success", is_error: false, usage });
  }
}
`,
    "utf8",
  );
  await chmod(executable, 0o755);

  const originalEntrypoint = process.env.CLAUDE_CODE_ENTRYPOINT;
  const originalOverride = process.env.PI_CLAUDE_CODE_ENTRYPOINT;
  delete process.env.PI_CLAUDE_CODE_ENTRYPOINT;
  try {
    for (const scenario of [
      { name: "default", host: undefined, override: undefined, expected: "pi-claude-code" },
      { name: "inherited host", host: "claude-desktop", override: undefined, expected: "pi-claude-code" },
      { name: "explicit override", host: "claude-desktop", override: "custom-pi", expected: "custom-pi" },
      { name: "empty override", host: "sdk-ts", override: "", expected: "pi-claude-code" },
    ]) {
      await t.test(scenario.name, async () => {
        if (scenario.host === undefined) delete process.env.CLAUDE_CODE_ENTRYPOINT;
        else process.env.CLAUDE_CODE_ENTRYPOINT = scenario.host;
        await writeFile(logPath, "", "utf8");

        const providerEnv: Record<string, string | undefined> = {
          PI_CLAUDE_CODE_COMMAND: executable,
          PI_CLAUDE_CODE_CONFIG_DIR: join(root, scenario.name),
          PI_CLAUDE_CODE_ENTRYPOINT: scenario.override,
        };
        const provider = createClaudeCodeProvider();
        const resolved = await provider.auth.apiKey!.resolve({
          ctx: { env: async (name) => providerEnv[name], fileExists: async () => false },
          signal: new AbortController().signal,
        });
        assert.equal(resolved?.env?.PI_CLAUDE_CODE_ENTRYPOINT, scenario.override);

        const status = await getClaudeStatus(claudeCodeEnv());
        assert.equal(status.loggedIn, true);
        const models = await discoverClaudeModels(claudeCodeEnv(), undefined, 5_000);
        assert.equal(models[0]?.value, "sonnet");
        await runClaudeRequest({
          model: "sonnet",
          frames: [{ type: "user", message: { role: "user", content: [{ type: "text", text: "test" }] } }],
          systemPrompt: "test",
          manifest: [],
          extraBody: {},
          toolNames: new Set(),
          env: resolved?.env,
          timeoutMs: 5_000,
        });

        const calls = (await readFile(logPath, "utf8")).trim().split("\n").map((line) =>
          JSON.parse(line) as { kind: string; entrypoint?: string },
        );
        assert.deepEqual([...new Set(calls.map((call) => call.kind))].sort(), ["auth", "discovery", "turn"]);
        for (const call of calls) assert.equal(call.entrypoint, scenario.expected, call.kind);
        assert.equal(process.env.CLAUDE_CODE_ENTRYPOINT, scenario.host);
        assert.equal(process.env.PI_CLAUDE_CODE_ENTRYPOINT, undefined);
        assert.equal(providerEnv.PI_CLAUDE_CODE_ENTRYPOINT, scenario.override);
      });
    }
  } finally {
    if (originalEntrypoint === undefined) delete process.env.CLAUDE_CODE_ENTRYPOINT;
    else process.env.CLAUDE_CODE_ENTRYPOINT = originalEntrypoint;
    if (originalOverride === undefined) delete process.env.PI_CLAUDE_CODE_ENTRYPOINT;
    else process.env.PI_CLAUDE_CODE_ENTRYPOINT = originalOverride;
    await rm(root, { recursive: true, force: true });
  }
});
