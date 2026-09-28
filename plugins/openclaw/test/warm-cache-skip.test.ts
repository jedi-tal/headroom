import { afterEach, describe, expect, it, vi } from "vitest";

const mocked = vi.hoisted(() => ({
  compress: vi.fn(),
  start: vi.fn(async () => "http://127.0.0.1:8787"),
  stop: vi.fn(async () => undefined),
  logger: { debug: vi.fn(), error: vi.fn(), info: vi.fn(), warn: vi.fn() },
}));

vi.mock("headroom-ai", () => ({ compress: mocked.compress }));

vi.mock("../src/proxy-manager.js", () => ({
  ProxyManager: class {
    start = mocked.start;
    stop = mocked.stop;
  },
  defaultLogger: mocked.logger,
}));

import { normalizeAgentMessages } from "../src/convert.js";
import { HeadroomContextEngine } from "../src/engine.js";

afterEach(() => {
  mocked.compress.mockReset();
});

// A history whose newest message is `ageMs` old: a user turn, a tool call, and a large tool result.
function history(ageMs: number) {
  const last = Date.now() - ageMs;
  return [
    { role: "user", content: [{ type: "text", text: "list the pods" }], timestamp: last - 2000 },
    {
      role: "assistant",
      content: [{ type: "toolCall", id: "t1", name: "exec", arguments: { command: "kubectl get pods" } }],
      api: "bedrock-converse-stream",
      provider: "amazon-bedrock",
      model: "claude",
      stopReason: "toolUse",
      timestamp: last - 1000,
    },
    { role: "toolResult", toolCallId: "t1", toolName: "exec", content: [{ type: "text", text: "x".repeat(4000) }], isError: false, timestamp: last },
  ];
}

// The proxy compresses the tool result and reports the sizes it saw.
function compressing(tokensBefore: number) {
  return async (messages: any[]) => ({
    compressed: true,
    tokensBefore,
    tokensAfter: tokensBefore - 800,
    tokensSaved: 800,
    messages: messages.map((m) => (m.role === "tool" ? { ...m, content: "[compressed]" } : m)),
  });
}

function engine(config: Record<string, unknown>) {
  const e = new HeadroomContextEngine(config as any);
  (e as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
  return e;
}

describe("skipCompressionWhenCacheWarmMs", () => {
  it("returns the history unchanged while the cache is warm", async () => {
    mocked.compress.mockImplementation(compressing(1000));
    const input = history(60_000);

    const result = await engine({ skipCompressionWhenCacheWarmMs: 300_000 }).assemble({ sessionId: "s", messages: input });

    expect(result.messages).toEqual(normalizeAgentMessages(input));
    expect(result.estimatedTokens).toBe(1000);
  });

  it("applies the compression once the cache has gone cold", async () => {
    mocked.compress.mockImplementation(compressing(1000));

    const result = await engine({ skipCompressionWhenCacheWarmMs: 300_000 }).assemble({ sessionId: "s", messages: history(600_000) });

    expect(result.messages[2].content).toEqual([{ type: "text", text: "[compressed]" }]);
  });

  it("still compresses a warm history that has to shrink to fit the token budget", async () => {
    mocked.compress.mockImplementation(compressing(1000));

    const result = await engine({ skipCompressionWhenCacheWarmMs: 300_000 }).assemble({
      sessionId: "s",
      messages: history(60_000),
      tokenBudget: 500,
    });

    expect(result.messages[2].content).toEqual([{ type: "text", text: "[compressed]" }]);
  });

  it("changes nothing when the option is 0", async () => {
    mocked.compress.mockImplementation(compressing(1000));

    const result = await engine({ skipCompressionWhenCacheWarmMs: 0 }).assemble({ sessionId: "s", messages: history(60_000) });

    expect(result.messages[2].content).toEqual([{ type: "text", text: "[compressed]" }]);
  });

  it("defaults to the 5-minute cache TTL on this deploy", async () => {
    mocked.compress.mockImplementation(compressing(1000));
    const warm = history(60_000);

    expect((await engine({}).assemble({ sessionId: "s", messages: warm })).messages).toEqual(normalizeAgentMessages(warm));
    expect((await engine({}).assemble({ sessionId: "s", messages: history(360_000) })).messages[2].content).toEqual([
      { type: "text", text: "[compressed]" },
    ]);
  });
});
