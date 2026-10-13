#!/usr/bin/env python
"""最小 MCP 客户端演示：用裸 JSON-RPC 消息与本项目 MCP 服务器对话。

流程（MCP 协议三步曲）：
  1. initialize   —— 握手：交换协议版本与能力
  2. tools/list   —— 客户端问"你有哪些工具"，服务器返回工具清单（名称/描述/参数模式）
  3. tools/call   —— 真正调用工具，拿回检索结果

这证明：任何支持 MCP 的客户端（Copilot/Claude Desktop）都能这样用你的知识库。

用法:
    python scripts/demo_mcp_client.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    # 启动服务器子进程：stdio 传输（stdin/stdout 就是通信管道）
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # 日志走 stderr，演示时不显示
        text=True,
        encoding="utf-8",
        cwd=str(_REPO_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    next_id = 0

    def send(method: str, params: dict | None = None, notification: bool = False) -> int:
        nonlocal next_id
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notification:
            next_id += 1
            msg["id"] = next_id
        proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        return next_id

    def recv(want_id: int) -> dict:
        """读服务器响应（跳过无关消息直到拿到目标 id 的响应）。"""
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("服务器意外关闭（看 stderr 日志排查）")
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("id") == want_id:
                return data

    try:
        # ── 第 1 步：initialize 握手 ────────────────────────────
        rid = send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "demo-client", "version": "0.1.0"},
        })
        init = recv(rid)
        info = init["result"]
        print("=" * 64)
        print("第 1 步 initialize 握手成功")
        print(f"  服务器: {info['serverInfo']['name']} v{info['serverInfo']['version']}")
        print(f"  协商出的协议版本: {info.get('protocolVersion')}")
        send("notifications/initialized", notification=True)

        # ── 第 2 步：tools/list 工具发现 ─────────────────────────
        rid = send("tools/list")
        tools = recv(rid)["result"]["tools"]
        print("-" * 64)
        print(f"第 2 步 tools/list：服务器有 {len(tools)} 个工具")
        for t in tools:
            print(f"  - {t['name']}: {t['description'][:60]}…")

        # ── 第 3 步：tools/call 真正调用 ─────────────────────────
        print("-" * 64)
        rid = send("tools/call", {
            "name": "list_collections",
            "arguments": {},
        })
        result = recv(rid)["result"]
        text = result["content"][0]["text"]
        print(f"第 3 步 tools/call → list_collections:")
        print("  " + text[:300].replace("\n", "\n  "))

        rid = send("tools/call", {
            "name": "query_knowledge_hub",
            "arguments": {"query": "重排序有哪些方案？", "top_k": 3, "collection": "learning_docs"},
        })
        result = recv(rid)["result"]
        print("-" * 64)
        print("第 4 步 tools/call → query_knowledge_hub:")
        for block in result["content"][:2]:
            if block.get("type") == "text":
                print("  " + block["text"][:400].replace("\n", "\n  ") + " …")

        print("=" * 64)
        print("MCP 服务器实测通过：任何 MCP 客户端都能这样使用你的知识库。")
        return 0
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
