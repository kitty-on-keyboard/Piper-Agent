// Drive the C++ server from the OFFICIAL MCP TypeScript SDK client.
//
// This is the direction that matters most: our conformance harness is something we
// wrote, so it can only find bugs we thought to look for. The reference client was
// written by someone who has never seen our code.

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const SERVER = process.argv[2];
if (!SERVER) {
  console.error("usage: node drive_our_server.mjs <path-to-server-binary>");
  process.exit(2);
}

let pass = 0;
let fail = 0;
function check(name, ok, detail = "") {
  if (ok) { pass++; console.log(`  PASS  ${name}${detail ? " — " + detail : ""}`); }
  else    { fail++; console.log(`  FAIL  ${name}${detail ? " — " + detail : ""}`); }
}

const transport = new StdioClientTransport({ command: SERVER, args: [], stderr: "ignore" });
const client = new Client({ name: "official-sdk-probe", version: "1.0.0" }, { capabilities: {} });

try {
  await client.connect(transport);

  const v = client.getServerVersion();
  const caps = client.getServerCapabilities();
  check("initialize handshake", true, `${v?.name} ${v?.version}`);
  check("server advertised capabilities", !!caps, JSON.stringify(caps));
  check("instructions delivered", !!client.getInstructions(),
        (client.getInstructions() || "").slice(0, 48) + "...");

  // --- tools --------------------------------------------------------------
  const tools = await client.listTools();
  check("tools/list", tools.tools.length === 4, `${tools.tools.length} tools`);
  check("tool schema present", !!tools.tools[0]?.inputSchema,
        tools.tools.map(t => t.name).join(", "));

  const echo = await client.callTool({ name: "echo", arguments: { text: "round trip" } });
  check("tools/call echo", echo.content?.[0]?.text === "round trip",
        JSON.stringify(echo.content?.[0]));

  const sum = await client.callTool({ name: "add", arguments: { a: 20, b: 22 } });
  check("tools/call structuredContent", sum.structuredContent?.sum === 42,
        JSON.stringify(sum.structuredContent));

  const boom = await client.callTool({ name: "always_fails", arguments: {} });
  check("tool failure is isError, not a protocol error", boom.isError === true,
        JSON.stringify(boom.content?.[0]?.text));

  // --- progress -----------------------------------------------------------
  let progressSeen = 0;
  const slow = await client.callTool(
    { name: "slow_count", arguments: { n: 3 } },
    undefined,
    { onprogress: (p) => { progressSeen++; } },
  );
  check("progress notifications during a call", progressSeen >= 3,
        `${progressSeen} progress events`);
  check("slow tool completed", /counted to 3/.test(slow.content?.[0]?.text ?? ""));

  // --- resources ----------------------------------------------------------
  const res = await client.listResources();
  check("resources/list", res.resources.length === 2, `${res.resources.length} resources`);
  const read = await client.readResource({ uri: "mem://readme" });
  check("resources/read", (read.contents?.[0]?.text ?? "").includes("static MCP resource"));

  // --- prompts ------------------------------------------------------------
  const prompts = await client.listPrompts();
  check("prompts/list", prompts.prompts.length === 1, prompts.prompts[0]?.name);
  const got = await client.getPrompt({
    name: "code_review",
    arguments: { language: "c++", code: "int main(){}" },
  });
  check("prompts/get", (got.messages?.[0]?.content?.text ?? "").includes("c++ code"));

  // --- completion ---------------------------------------------------------
  const comp = await client.complete({
    ref: { type: "ref/prompt", name: "code_review" },
    argument: { name: "language", value: "c" },
  });
  check("completion/complete", (comp.completion?.values ?? []).includes("c++"),
        JSON.stringify(comp.completion?.values));

  // --- utilities ----------------------------------------------------------
  await client.ping();
  check("ping", true);

  await client.setLoggingLevel("debug");
  check("logging/setLevel", true);

  // --- error handling -----------------------------------------------------
  try {
    await client.callTool({ name: "no_such_tool", arguments: {} });
    check("unknown tool rejected", false, "call unexpectedly succeeded");
  } catch (e) {
    check("unknown tool rejected", true, `code ${e.code}`);
  }

  await client.close();
} catch (err) {
  console.log(`  FAIL  fatal: ${err?.stack || err}`);
  fail++;
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
