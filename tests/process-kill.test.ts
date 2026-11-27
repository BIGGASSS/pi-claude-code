import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { KILL_GRACE_MS, killProcessTree } from "../src/claude-cli.ts";

const posixOnly = { skip: process.platform === "win32" ? "POSIX process groups only" : false };

function exited(child: ChildProcess): Promise<{ code: number | null; signal: NodeJS.Signals | null }> {
  return new Promise((resolve) => child.once("exit", (code, signal) => resolve({ code, signal })));
}

function ready(child: ChildProcess): Promise<void> {
  return new Promise((resolve) => child.stdout!.once("data", () => resolve()));
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function spawnWorker(script: string): ChildProcess {
  return spawn(process.execPath, ["-e", script], { detached: true, stdio: ["ignore", "pipe", "ignore"] });
}

test("lets Claude Code clean up on exit before force-killing (issue #50)", posixOnly, async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "pi-claude-code-kill-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const socket = join(dir, "worker.sock");
  // Stands in for Claude Code, which deletes its cc-socks socket on a normal exit.
  const child = spawnWorker(`
    const fs = require("node:fs");
    fs.writeFileSync(${JSON.stringify(socket)}, "");
    process.on("SIGTERM", () => { fs.rmSync(${JSON.stringify(socket)}); process.exit(0); });
    setInterval(() => {}, 1000);
    process.stdout.write("ready\\n");
  `);
  await ready(child);
  const exit = exited(child);
  killProcessTree(child);
  assert.deepEqual(await exit, { code: 0, signal: null });
  assert.equal(existsSync(socket), false);
});

test("force-kills a worker and its tools when they ignore SIGTERM", posixOnly, async () => {
  // The worker and a grandchild in its process group both ignore SIGTERM.
  const child = spawnWorker(`
    const { spawn } = require("node:child_process");
    process.on("SIGTERM", () => {});
    const tool = spawn(process.execPath, ["-e", "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000)"], { stdio: "ignore" });
    setInterval(() => {}, 1000);
    setTimeout(() => process.stdout.write(tool.pid + "\\n"), 200);
  `);
  const toolPid = await new Promise<number>((resolve) =>
    child.stdout!.once("data", (chunk: Buffer) => resolve(Number(chunk.toString().trim()))),
  );
  assert.ok(alive(toolPid));
  const started = Date.now();
  const exit = exited(child);
  killProcessTree(child);
  assert.deepEqual(await exit, { code: null, signal: "SIGKILL" });
  assert.ok(Date.now() - started >= KILL_GRACE_MS - 100);
  for (let i = 0; i < 50 && alive(toolPid); i++) await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(alive(toolPid), false);
});
