#!/usr/bin/env python3
"""json_deep_merge.py — 高性能、可回滚的 JSON 深度合并器（仅标准库，单文件）

用法
----
    # 命令行：依次合并多个 JSON 文件，结果输出到 stdout
    python3 json_deep_merge.py base.json override1.json override2.json
    python3 json_deep_merge.py --array-mode concat --key-conflict keep a.json b.json
    python3 json_deep_merge.py --demo          # 运行内置示例（合并 + 回滚 + 基准）

    # 作为库：
    from json_deep_merge import Merger
    m = Merger(base_dict, key_conflict="override", array_mode="replace")
    rec = m.merge(incoming_dict)   # 返回 ChangeRecord（增量变更记录）
    m.undo()                       # 撤销最近一次合并
    m.rollback_to(rec.id)          # 回滚到某次合并之前（含该次，LIFO）

策略说明（取舍）
----------------
key_conflict（同名标量键冲突）：
    "override"  后者覆盖前者（默认）。适合"配置层层覆盖"场景，最常用。
    "keep"      保留前者。适合"默认值兜底、先写优先"场景。
    "error"     抛 ConflictError。适合合并前要求键集合互斥、冲突即数据错误的场景。
array_mode（同键两边都是数组）：
    "replace"   整体替换（默认）。语义简单、可预测，回滚只需存旧引用。
    "concat"    拼接。适合白名单/插件列表类配置，但回滚需记录原长度，
                且重复元素不去重（去重要 O(n) 扫描且破坏顺序语义，不做）。

性能设计（用什么换什么）
-----------------------
1. 增量复用 / 结构化共享：incoming 中 base 不存在的子树【整体按引用挂载】，
   O(1) 完成，绝不逐叶拷贝；合并复杂度只与"两边重叠的路径"成正比，
   与 base 文档总大小（几万键）无关。
   —— 用【别名风险】换【时间】：合并后若调用方继续修改 incoming 的子树，
   会反映到结果里。需要隔离时构造 Merger(copy_incoming=True)，
   代价是退化为 O(incoming 大小) 的深拷贝。
2. 变更日志即路径索引：每个 ChangeRecord 只记录被改动的节点路径与旧值，
   新增子树只记一条 "add"（而非每叶一条），回滚代价 O(变更节点数)，
   不整树重扫、不整树快照。
   —— 用【少量内存】（日志与变更量成正比）换【免快照的撤销能力】。
3. 显式栈迭代代替递归：深层嵌套不会触发 Python 递归深度限制
   （注意：json.loads 解析超深 JSON 本身仍有递归限制，必要时自行
   sys.setrecursionlimit）。

错误语义
--------
- 类型冲突：同键一边对象/数组/标量的"种类"不一致（尤其对象 vs 数组），
  抛 TypeConflictError，带完整路径。容器与标量互相覆盖风险太大，不允许。
- 循环引用：输入对象图存在环时抛 CycleError（按祖先链检测，DAG 共享不误报）。
- 原子性：合并中途抛错时，本次已应用的操作会自动反向撤销，文档保持合并前状态。
- 回滚顺序：undo/rollback_to 必须按 LIFO（后进先出）进行；跳过中间记录
  单独撤销在路径重叠时无意义，故不支持。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = [
    "Merger",
    "ChangeRecord",
    "Op",
    "MergeError",
    "ConflictError",
    "TypeConflictError",
    "CycleError",
]

# 策略常量
OVERRIDE, KEEP, ERROR = "override", "keep", "error"
REPLACE, CONCAT = "replace", "concat"

_MISSING = object()  # 哨兵：区分"旧值是 None"与"键原本不存在"


class MergeError(Exception):
    """所有合并错误的基类。"""


class ConflictError(MergeError):
    """key_conflict="error" 策略下出现同名标量键冲突。"""


class TypeConflictError(MergeError):
    """同键两侧 JSON 种类（对象/数组/标量）不一致。"""


class CycleError(MergeError):
    """输入对象图存在循环引用。"""


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "scalar"


def _fmt_path(path: Tuple[str, ...]) -> str:
    return "$" + "".join(f"[{p!r}]" for p in path)


def _check_cycles(root: Any) -> None:
    """迭代 DFS 检测循环引用（只按祖先链判定，DAG 共享子树不算环）。"""
    on_path: set[int] = set()
    stack: List[Tuple[str, Any]] = [("enter", root)]
    while stack:
        tag, node = stack.pop()
        if tag == "exit":
            on_path.discard(id(node))
            continue
        if not isinstance(node, (dict, list)):
            continue
        nid = id(node)
        if nid in on_path:
            raise CycleError("检测到循环引用，无法合并（输入必须是树状 JSON 对象图）")
        on_path.add(nid)
        stack.append(("exit", node))
        children = node.values() if isinstance(node, dict) else node
        for child in children:
            stack.append(("enter", child))


@dataclass
class Op:
    """一条可逆操作。路径指向被修改的键本身（父路径 + 最后一节）。"""

    kind: str  # "add"（新增键）| "set"（覆盖旧值）| "concat"（数组拼接）
    path: Tuple[str, ...]
    old: Any = _MISSING  # "set"：被覆盖的旧值（引用，不拷贝）
    old_len: int = 0  # "concat"：拼接前数组长度


@dataclass
class ChangeRecord:
    """一次 merge() 产生的增量变更记录，可用于回滚。"""

    id: int
    ops: List[Op] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    note: str = ""

    def __len__(self) -> int:
        return len(self.ops)


class Merger:
    """绑定一份基准文档，逐次合并并维护回滚历史。"""

    def __init__(
        self,
        document: Dict[str, Any],
        *,
        key_conflict: str = OVERRIDE,
        array_mode: str = REPLACE,
        copy_incoming: bool = False,
    ) -> None:
        if not isinstance(document, dict):
            raise TypeConflictError("基准文档顶层必须是 JSON 对象（dict）")
        if key_conflict not in (OVERRIDE, KEEP, ERROR):
            raise ValueError(f"未知 key_conflict 策略: {key_conflict!r}")
        if array_mode not in (REPLACE, CONCAT):
            raise ValueError(f"未知 array_mode 策略: {array_mode!r}")
        _check_cycles(document)
        self.document = document
        self.key_conflict = key_conflict
        self.array_mode = array_mode
        self.copy_incoming = copy_incoming
        self._history: List[ChangeRecord] = []
        self._next_id = 1

    @property
    def history(self) -> List[ChangeRecord]:
        return list(self._history)

    # ------------------------------------------------------------------ merge
    def merge(self, incoming: Dict[str, Any], *, note: str = "") -> ChangeRecord:
        """把 incoming 深度合并进文档，返回增量变更记录。出错时自动回滚本次。"""
        if not isinstance(incoming, dict):
            raise TypeConflictError("合并输入顶层必须是 JSON 对象（dict）")
        _check_cycles(incoming)
        if self.copy_incoming:
            incoming = json.loads(json.dumps(incoming))  # 纯 JSON 安全的深拷贝

        record = ChangeRecord(id=self._next_id, note=note)
        self._next_id += 1
        try:
            self._apply(self.document, incoming, record)
        except MergeError:
            self._undo_ops(record)  # 原子性：撤销本次已应用的部分
            raise
        self._history.append(record)
        return record

    def _apply(self, base: Dict[str, Any], incoming: Dict[str, Any], rec: ChangeRecord) -> None:
        # 显式栈：只下钻"两边都是对象"的重叠路径，与 base 总键数无关
        stack: List[Tuple[Dict[str, Any], Dict[str, Any], Tuple[str, ...]]] = [
            (base, incoming, ())
        ]
        while stack:
            bnode, inode, path = stack.pop()
            for key, value in inode.items():
                if key not in bnode:
                    # 整棵新子树按引用挂载：O(1)，一条 add 记录即可撤销
                    bnode[key] = value
                    rec.ops.append(Op("add", path + (key,)))
                    continue
                current = bnode[key]
                ckind, vkind = _kind(current), _kind(value)
                if ckind != vkind:
                    raise TypeConflictError(
                        f"类型冲突 @ {_fmt_path(path + (key,))}: "
                        f"现有为 {ckind}，合并输入为 {vkind}，不允许跨种类覆盖"
                    )
                if ckind == "object":
                    stack.append((current, value, path + (key,)))
                elif ckind == "array":
                    if self.array_mode == CONCAT:
                        rec.ops.append(Op("concat", path + (key,), old_len=len(current)))
                        current.extend(value)
                    else:  # replace
                        rec.ops.append(Op("set", path + (key,), old=current))
                        bnode[key] = value
                else:  # 标量冲突，按策略
                    if self.key_conflict == OVERRIDE:
                        rec.ops.append(Op("set", path + (key,), old=current))
                        bnode[key] = value
                    elif self.key_conflict == ERROR:
                        raise ConflictError(
                            f"键冲突 @ {_fmt_path(path + (key,))}: "
                            f"{current!r} vs {value!r}（key_conflict='error'）"
                        )
                    # KEEP：不动，也不记录

    # --------------------------------------------------------------- rollback
    def undo(self) -> ChangeRecord:
        """撤销最近一次合并，返回被撤销的记录。"""
        if not self._history:
            raise MergeError("没有可撤销的合并记录")
        record = self._history.pop()
        self._undo_ops(record)
        return record

    def rollback_to(self, record_id: int) -> int:
        """回滚到 record_id 对应合并【之前】的状态（含该次，按 LIFO 依次撤销）。

        返回实际撤销的记录条数。
        """
        ids = [r.id for r in self._history]
        if record_id not in ids:
            raise MergeError(f"记录 #{record_id} 不在历史中（现存: {ids}）")
        count = 0
        while self._history and self._history[-1].id >= record_id:
            self.undo()
            count += 1
        return count

    def _undo_ops(self, record: ChangeRecord) -> None:
        for op in reversed(record.ops):
            if op.kind == "add":
                parent = self._resolve(op.path[:-1])
                del parent[op.path[-1]]
            elif op.kind == "set":
                parent = self._resolve(op.path[:-1])
                parent[op.path[-1]] = op.old
            else:  # concat：截掉本次拼接的尾部
                lst = self._resolve(op.path)
                del lst[op.old_len :]

    def _resolve(self, path: Tuple[str, ...]) -> Any:
        node = self.document
        for key in path:
            node = node[key]
        return node


# --------------------------------------------------------------------- demo
def _demo() -> None:
    print("=== 1. 基本合并（override + replace）===")
    base = {
        "server": {"host": "0.0.0.0", "port": 80, "opts": {"tls": False}},
        "tags": ["a", "b"],
        "name": "svc",
    }
    m = Merger(base)
    r1 = m.merge({"server": {"port": 8080, "opts": {"tls": True, "ca": "/etc/ca"}}},
                 note="调端口+开TLS")
    r2 = m.merge({"tags": ["x", "y"], "extra": {"deep": {"a": {"b": 1}}}},
                 note="替换tags+新增深层子树")
    print(json.dumps(m.document, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"历史: {[(r.id, len(r), r.note) for r in m.history]}")

    print("\n=== 2. 回滚到 r1 之前（撤销 r1、r2）===")
    m.rollback_to(r1.id)
    print(json.dumps(m.document, ensure_ascii=False, sort_keys=True))

    print("\n=== 3. 数组拼接 + 保留前者策略 ===")
    m2 = Merger({"list": [1, 2], "k": "first"}, array_mode=CONCAT, key_conflict=KEEP)
    m2.merge({"list": [3, 4], "k": "second"})
    print(json.dumps(m2.document, ensure_ascii=False, sort_keys=True))
    m2.undo()
    print("undo 后:", json.dumps(m2.document, ensure_ascii=False, sort_keys=True))

    print("\n=== 4. 错误路径：冲突 / 类型冲突 / 循环引用（均不破坏文档）===")
    try:
        Merger({"a": 1}, key_conflict=ERROR).merge({"a": 2})
    except ConflictError as e:
        print("ConflictError:", e)
    doc = {"cfg": {"x": 1}}
    try:
        Merger(doc).merge({"cfg": [1, 2]})
    except TypeConflictError as e:
        print("TypeConflictError:", e)
    print("类型冲突后文档保持:", json.dumps(doc, sort_keys=True))
    cyc: Dict[str, Any] = {"self": None}
    cyc["self"] = cyc
    try:
        Merger({}).merge(cyc)
    except CycleError as e:
        print("CycleError:", e)

    print("\n=== 5. 基准：5 万键深层文档，合并 50 份各改 200 键 ===")
    big: Dict[str, Any] = {}
    for i in range(500):  # 500 * 100 = 50,000 键
        big[f"mod{i:04d}"] = {f"k{j:04d}": j for j in range(100)}
    bm = Merger(big)
    t0 = time.perf_counter()
    for n in range(50):
        bm.merge({f"mod{n:04d}": {f"k{j:04d}": -1 for j in range(200)}},
                 note=f"batch{n}")
    t1 = time.perf_counter()
    bm.rollback_to(bm.history[0].id)  # 全部撤销
    t2 = time.perf_counter()
    ok = all(big[f"mod{i:04d}"][f"k{j:04d}"] == j for i in (0, 250, 499) for j in (0, 99))
    print(f"50 次合并耗时 {t1 - t0:.3f}s，全部回滚耗时 {t2 - t1:.3f}s，"
          f"回滚后数据校验: {'OK' if ok else 'FAIL'}")


def main(argv: List[str]) -> int:
    args = list(argv)
    if "--demo" in args or not args:
        _demo()
        return 0
    key_conflict, array_mode = OVERRIDE, REPLACE
    files: List[str] = []
    it = iter(range(len(args)))
    i = 0
    while i < len(args):
        if args[i] == "--key-conflict":
            key_conflict = args[i + 1]
            i += 2
        elif args[i] == "--array-mode":
            array_mode = args[i + 1]
            i += 2
        else:
            files.append(args[i])
            i += 1
    if not files:
        print("用法: json_deep_merge.py [--key-conflict override|keep|error] "
              "[--array-mode replace|concat] FILE...", file=sys.stderr)
        return 2
    with open(files[0], encoding="utf-8") as f:
        merged = json.load(f)
    merger = Merger(merged, key_conflict=key_conflict, array_mode=array_mode)
    for path in files[1:]:
        with open(path, encoding="utf-8") as f:
            merger.merge(json.load(f), note=path)
    json.dump(merger.document, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
