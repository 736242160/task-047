"""json_deep_merge.py — 单文件 JSON 深度合并器（仅标准库）。

功能
----
1. 深度合并：对象递归合并；数组与标量按可配置策略处理。
2. 同名标量键冲突策略：OVERRIDE（后者覆盖）/ KEEP_FIRST（保留前者）/ ERROR（抛错）。
3. 数组策略：REPLACE（整体替换）/ CONCAT（拼接）。
4. 类型冲突（同键一边对象、一边数组/标量）抛 TypeConflictError。
5. 循环引用抛 CircularReferenceError；合并中途出错自动撤销本次部分变更（原子性）。
6. 每次 merge 返回 MergeRecord（增量变更日志），支持 LIFO 撤销或回滚到指定记录。

性能设计（用什么换什么）
----------------------
- 合并代价 = O(本次输入片段大小)，只下钻片段里出现的路径，不回溯整棵树。
  换来的是：大文档上反复合并小片段时，单次耗时与文档总大小无关。
- 回滚代价 = O(该次变更条数)。每条 _Change 直接持有父容器引用（即"路径索引"），
  回滚时 O(1) 定位修改点，无需重新扫描或 diff 整树。
  换来的是：撤销必须按 LIFO 顺序（跳过中间记录撤销是不安全的，引擎会强制）。
- 新子树按引用挂入，不做深拷贝，省去 O(子树) 拷贝。
  代价：约定输入片段合并后不再被外部修改（json.loads 的产物天然满足）。
- 不维护全量路径表，只为被触及的路径建索引（变更记录本身）。
  全量表内存 O(总键数) 且每次合并都要同步维护，收益抵不上成本。
- 合并与环检测均用显式栈实现，深层嵌套不受 Python 递归深度限制。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------- 错误类型

class MergeError(Exception):
    """合并相关错误的基类。"""


class ConflictError(MergeError):
    """同名标量键冲突且策略为 ERROR 时抛出。"""


class TypeConflictError(MergeError):
    """同键两侧类型不可调和（如一边是对象、一边是数组或标量）。"""


class CircularReferenceError(MergeError):
    """输入片段存在循环引用。"""


class RollbackError(MergeError):
    """回滚请求非法（没有历史、记录不存在或已撤销）。"""


# ---------------------------------------------------------------- 策略枚举

class ConflictStrategy(Enum):
    OVERRIDE = "override"      # 后者覆盖前者（默认，适合"配置覆盖"场景）
    KEEP_FIRST = "keep_first"  # 保留前者（适合"默认值填充"场景）
    ERROR = "error"            # 抛 ConflictError（适合"配置不允许歧义"场景）


class ArrayStrategy(Enum):
    REPLACE = "replace"  # 数组整体替换（默认，语义最不易误解）
    CONCAT = "concat"    # 数组拼接，后者接在前者之后（适合白名单类累加配置）


# ---------------------------------------------------------------- 内部结构

_MISSING = object()  # 哨兵：表示"该键在合并前不存在"


def _fmt_path(path: Tuple[Any, ...]) -> str:
    out = ["$"]
    for key in path:
        if isinstance(key, str) and key.isidentifier():
            out.append(f".{key}")
        else:
            out.append(f"[{key!r}]")
    return "".join(out)


@dataclass(eq=False)
class _Change:
    """一条增量变更。parent 是对父容器的直接引用——这就是路径索引：

    回滚时无需从根沿路径查找，O(1) 即可定位修改点。
    """
    parent: Any
    key: Any
    path: Tuple[Any, ...]
    old: Any = _MISSING
    new: Any = None


@dataclass(eq=False)
class MergeRecord:
    """一次合并的增量变更记录，可用于回滚。"""
    seq: int
    description: str
    timestamp: float
    changes: List[_Change] = field(default_factory=list)
    undone: bool = False

    def __repr__(self) -> str:
        return (f"<MergeRecord #{self.seq} {self.description!r} "
                f"changes={len(self.changes)} undone={self.undone}>")


def _check_acyclic(root: Any) -> None:
    """用显式栈检测输入片段中的循环引用（共享子树的 DAG 是允许的）。"""
    stack: List[Tuple[str, Any, Tuple[Any, ...]]] = [("enter", root, ())]
    active: set = set()
    while stack:
        event, node, path = stack.pop()
        if event == "exit":
            active.discard(id(node))
            continue
        if isinstance(node, dict):
            children = node.items()
        elif isinstance(node, list):
            children = enumerate(node)
        else:
            continue
        if id(node) in active:
            raise CircularReferenceError(f"检测到循环引用：{_fmt_path(path)}")
        active.add(id(node))
        stack.append(("exit", node, path))
        for key, value in children:
            stack.append(("enter", value, path + (key,)))


# ---------------------------------------------------------------- 合并器

class Merger:
    """维护一份合并中的文档，支持增量合并与按记录回滚。"""

    def __init__(self,
                 conflict: ConflictStrategy = ConflictStrategy.OVERRIDE,
                 array: ArrayStrategy = ArrayStrategy.REPLACE,
                 initial: Optional[dict] = None):
        self._conflict = conflict
        self._array = array
        self._doc: dict = {}
        self._history: List[MergeRecord] = []
        self._seq = 0
        if initial:
            self.merge(initial, description="initial")

    @property
    def document(self) -> dict:
        return self._doc

    @property
    def history(self) -> List[MergeRecord]:
        return list(self._history)

    # ------------------------------------------------------------ 合并

    def merge(self, fragment: dict, description: str = "") -> MergeRecord:
        """把 fragment 深度合并进当前文档，返回变更记录。

        若合并中途抛错（类型冲突 / 策略 ERROR），本次已应用的部分变更
        会被自动撤销，文档保持合并前状态（原子性）。
        """
        if not isinstance(fragment, dict):
            raise MergeError(
                f"顶层输入必须是 JSON 对象（dict），得到 {type(fragment).__name__}")
        _check_acyclic(fragment)  # 只校验输入片段，不重扫已有文档
        self._seq += 1
        record = MergeRecord(self._seq, description, time.time())
        try:
            self._merge_dict(self._doc, fragment, (), record)
        except MergeError:
            self._undo_record(record)
            raise
        self._history.append(record)
        return record

    def _merge_dict(self, target: dict, src: dict,
                    path: Tuple[Any, ...], record: MergeRecord) -> None:
        # 显式栈下钻：只访问 src 里出现的路径，与文档其余部分无关
        stack: List[Tuple[dict, dict, Tuple[Any, ...]]] = [(target, src, path)]
        while stack:
            tgt, fragment, cpath_base = stack.pop()
            for key, sval in fragment.items():
                cpath = cpath_base + (key,)
                if key not in tgt:
                    tgt[key] = sval  # 按引用挂入，不做深拷贝
                    record.changes.append(_Change(tgt, key, cpath, _MISSING, sval))
                    continue
                tval = tgt[key]
                t_is_dict, s_is_dict = isinstance(tval, dict), isinstance(sval, dict)
                t_is_list, s_is_list = isinstance(tval, list), isinstance(sval, list)

                if t_is_dict or s_is_dict:
                    if t_is_dict and s_is_dict:
                        stack.append((tval, sval, cpath))
                    else:
                        raise TypeConflictError(
                            f"类型冲突：{_fmt_path(cpath)} 一侧是对象，"
                            f"另一侧是 {type(sval if t_is_dict else tval).__name__}")
                elif t_is_list or s_is_list:
                    if t_is_list and s_is_list:
                        if self._array is ArrayStrategy.CONCAT:
                            new_val = tval + sval  # 生成新列表，旧列表留给回滚
                        else:
                            new_val = sval
                        tgt[key] = new_val
                        record.changes.append(_Change(tgt, key, cpath, tval, new_val))
                    else:
                        raise TypeConflictError(
                            f"类型冲突：{_fmt_path(cpath)} 一侧是数组，"
                            f"另一侧是 {type(sval if t_is_list else tval).__name__}")
                else:
                    # 双方都是标量
                    if type(tval) is type(sval) and tval == sval:
                        continue  # 值未变，不产生变更记录
                    if self._conflict is ConflictStrategy.OVERRIDE:
                        tgt[key] = sval
                        record.changes.append(_Change(tgt, key, cpath, tval, sval))
                    elif self._conflict is ConflictStrategy.KEEP_FIRST:
                        continue
                    else:
                        raise ConflictError(
                            f"标量冲突：{_fmt_path(cpath)} "
                            f"{tval!r} vs {sval!r}（策略为 ERROR）")

    # ------------------------------------------------------------ 回滚

    def rollback(self, record: Optional[MergeRecord] = None) -> List[MergeRecord]:
        """撤销最近一次合并；或回滚到指定记录（含它在内的所有更新记录）。

        撤销严格按 LIFO 顺序进行——这是"变更记录持有父容器引用"设计的
        前提。返回本次实际撤销的记录列表（新的在前）。
        """
        if not self._history:
            raise RollbackError("没有可回滚的合并记录")
        if record is None:
            target = self._history[-1]
        else:
            if not any(r is record for r in self._history):
                raise RollbackError(f"记录不存在或已撤销：{record!r}")
            target = record
        undone: List[MergeRecord] = []
        while self._history:
            rec = self._history.pop()
            self._undo_record(rec)
            undone.append(rec)
            if rec is target:
                break
        return undone

    def _undo_record(self, record: MergeRecord) -> None:
        for change in reversed(record.changes):
            if change.old is _MISSING:
                change.parent.pop(change.key, None)  # 合并前不存在 -> 删除
            else:
                change.parent[change.key] = change.old  # 恢复旧值
        record.undone = True


# ---------------------------------------------------------------- 演示与自检

def _demo() -> None:
    print("=== 1. 基本合并（OVERRIDE + 数组 REPLACE）===")
    m = Merger(ConflictStrategy.OVERRIDE, ArrayStrategy.REPLACE)
    r1 = m.merge({
        "app": {"name": "shop", "replicas": 1},
        "db": {"host": "127.0.0.1", "port": 5432,
               "options": {"ssl": False, "pool": 4}},
        "tags": ["base"],
    }, description="基础配置")
    snapshot = json.loads(json.dumps(m.document))  # 合并前的深拷贝快照

    r2 = m.merge({
        "app": {"replicas": 3},
        "db": {"host": "db.prod.internal", "options": {"ssl": True}},
        "tags": ["prod", "eu"],
        "extra": {"feature_x": True},
    }, description="生产环境覆盖")

    print(f"记录: {r1!r}")
    print(f"记录: {r2!r}")
    print("合并结果:")
    print(json.dumps(m.document, ensure_ascii=False, indent=2, sort_keys=True))
    assert m.document["db"]["options"] == {"ssl": True, "pool": 4}  # 深层递归合并
    assert m.document["tags"] == ["prod", "eu"]                     # 数组整体替换

    print("\n=== 2. 按记录回滚 r2（生产环境覆盖）===")
    m.rollback(r2)
    print(json.dumps(m.document, ensure_ascii=False, indent=2, sort_keys=True))
    assert m.document == snapshot, "回滚后应精确恢复到合并前状态"
    print(">> 回滚校验通过：文档与合并前快照完全一致")

    print("\n=== 3. 类型冲突（对象 vs 数组）===")
    try:
        m.merge({"db": {"options": ["not", "a", "dict"]}})
    except TypeConflictError as exc:
        print(f">> 按预期报错: {exc}")
    assert m.document == snapshot, "出错后文档应保持不变（原子性）"
    print(">> 原子性校验通过：失败合并未留下残留修改")

    print("\n=== 4. 循环引用 ===")
    cyclic: dict = {"a": {}}
    cyclic["a"]["self"] = cyclic
    try:
        m.merge(cyclic)
    except CircularReferenceError as exc:
        print(f">> 按预期报错: {exc}")

    print("\n=== 5. 策略：KEEP_FIRST / ERROR / 数组 CONCAT ===")
    m_keep = Merger(ConflictStrategy.KEEP_FIRST, ArrayStrategy.CONCAT)
    m_keep.merge({"a": 1, "list": [1, 2]}, "first")
    m_keep.merge({"a": 99, "list": [3], "b": 2}, "second")
    print(f"KEEP_FIRST+CONCAT 结果: {m_keep.document}")
    assert m_keep.document == {"a": 1, "list": [1, 2, 3], "b": 2}

    m_err = Merger(ConflictStrategy.ERROR)
    m_err.merge({"a": 1})
    try:
        m_err.merge({"a": 2})
    except ConflictError as exc:
        print(f">> 按预期报错: {exc}")
    assert m_err.document == {"a": 1}

    print("\n=== 6. 性能冒烟：5 万键文档上反复增量合并 ===")
    big = {f"group{i:04d}": {f"key{j:03d}": j for j in range(100)}
           for i in range(500)}  # 50_000 个叶子键
    bench = Merger()
    t0 = time.perf_counter()
    bench.merge(big, "大文档")
    t1 = time.perf_counter()
    records = []
    for i in range(200):  # 模拟每天几十份小配置持续合并
        records.append(bench.merge(
            {f"group{i:04d}": {"key000": -1, "key001": -2}},
            description=f"增量{i}"))
    t2 = time.perf_counter()
    bench.rollback(records[100])  # 回滚掉后 100 次增量
    t3 = time.perf_counter()
    total_keys = sum(len(v) for v in bench.document.values())
    print(f"初次合并 5 万键: {(t1 - t0) * 1e3:.1f} ms")
    print(f"200 次增量合并:  {(t2 - t1) * 1e3:.1f} ms "
          f"(平均 {(t2 - t1) / 200 * 1e6:.0f} µs/次，与文档总大小无关)")
    print(f"回滚 100 次记录: {(t3 - t2) * 1e3:.1f} ms（只重放变更，不扫树）")
    assert total_keys == 50_000
    assert bench.document["group0099"]["key000"] == -1   # 前 100 次保留
    assert bench.document["group0100"]["key000"] == 0    # 后 100 次已撤销
    print(">> 性能与回滚正确性校验通过")


if __name__ == "__main__":
    _demo()
