"""批次换装顺序规划(只读): 以空转子为起点, 精确求总装卸操作数最少的执行次序。

成本模型:
- 初始转子为空; 第一批装入该方案全部试管, 每支装入计 1 次操作;
- 从方案 A 切换到方案 B: 先卸下所有"不在 B 的原槽位"的试管(B 中无此编号
  或槽位不同均须卸下), 再装入 B 中尚未就位的试管; 每支卸下/装入各计 1 次;
- 设 keep(A,B) 为两方案中同编号且同槽位的试管数, 则切换成本
  cost(A→B) = |A| + |B| - 2·keep(A,B), 仅取决于相邻两个方案。

精确优化: k∈[2,10] 条方案, 用 Held-Karp 动态规划在 O(2^k·k²) 内精确求总
操作数最小的排列; 同成本时按方案 ID 序列升序(字典序)裁决。最优子结构:
总成本 = 前缀成本 + 后续成本, 后续成本只依赖前缀末尾方案, 故每个
(子集, 末尾) 状态只需保留 (最小成本, 同成本下 ID 序列最小) 的前缀。

兼容性(不满足则拒绝, 抛 ValueError):
- 各方案转子半径一致;
- 槽位编号集合、逐槽角度与启用状态一致;
- 同编号试管在各方案中的质量一致(试管清单本身允许不同)。

只读: 不修改任何输入方案, 规划结果不落库。
"""
from __future__ import annotations

from typing import Any

_NUM_TOL = 1e-9
MIN_PLANS = 2
MAX_PLANS = 10


# ---------------------------------------------------------------------------
# 方案特征提取
# ---------------------------------------------------------------------------
def _slot_signature(plan: dict[str, Any]) -> dict[int, tuple[float, bool]]:
    """槽位编号 -> (角度, 启用状态), 取自方案保存时的请求(已归一化)。"""
    req = plan.get("request") or {}
    return {
        int(s["index"]): (float(s.get("angle", 0.0)), bool(s.get("enabled", True)))
        for s in req.get("slots") or []
    }


def _tube_masses(plan: dict[str, Any]) -> dict[int, float]:
    req = plan.get("request") or {}
    return {int(t["id"]): float(t["mass"]) for t in req.get("tubes") or []}


def _placement_map(plan: dict[str, Any]) -> dict[int, int]:
    """试管编号 -> 落槽槽位(取自求解结果的实际摆放)。"""
    out: dict[int, int] = {}
    for p in (plan.get("result") or {}).get("placements") or []:
        if p.get("slot") is None:
            continue
        out[int(p["tube_id"])] = int(p["slot"])
    return out


def _label(plan: dict[str, Any]) -> str:
    return f"方案 #{plan.get('id')}"


# ---------------------------------------------------------------------------
# 兼容性校验
# ---------------------------------------------------------------------------
def validate_compatible(plans: list[dict[str, Any]]) -> dict[int, float]:
    """数量/重复/几何/质量兼容性校验, 不满足抛 ValueError; 通过则返回合并质量表。"""
    if not (MIN_PLANS <= len(plans) <= MAX_PLANS):
        raise ValueError(
            f"批次换装需选择 {MIN_PLANS}~{MAX_PLANS} 条已保存方案"
            f"(当前 {len(plans)} 条)。"
        )
    ids = [int(p.get("id")) for p in plans]
    if len(set(ids)) != len(ids):
        raise ValueError("所选方案存在重复 ID, 请去除重复后再规划。")

    base = plans[0]
    base_radius = float((base.get("request") or {}).get("radius_mm", 100.0))
    base_slots = _slot_signature(base)
    masses = _tube_masses(base)
    mass_src = {tid: _label(base) for tid in masses}
    for plan in plans[1:]:
        req = plan.get("request") or {}
        radius = float(req.get("radius_mm", 100.0))
        if abs(radius - base_radius) > _NUM_TOL:
            raise ValueError(
                f"{_label(plan)} 的转子半径 {radius:g}mm 与 {_label(base)} 的 "
                f"{base_radius:g}mm 不一致, 批次换装要求所有方案半径相同。"
            )
        sig = _slot_signature(plan)
        if set(sig) != set(base_slots):
            raise ValueError(
                f"{_label(plan)} 的槽位编号集合与 {_label(base)} 不一致, "
                f"批次换装要求槽位数与槽位编号完全相同。"
            )
        for idx in sorted(sig):
            ang, enabled = sig[idx]
            base_ang, base_enabled = base_slots[idx]
            if abs(ang - base_ang) > _NUM_TOL:
                raise ValueError(
                    f"{_label(plan)} 槽位 {idx} 的角度 {ang:g}° 与 {_label(base)} 的 "
                    f"{base_ang:g}° 不一致, 批次换装要求槽位角度完全相同。"
                )
            if enabled != base_enabled:
                raise ValueError(
                    f"{_label(plan)} 槽位 {idx} 的启用状态与 {_label(base)} 不一致, "
                    f"批次换装要求槽位启用状态完全相同。"
                )
        for tid, m in _tube_masses(plan).items():
            if tid in masses and abs(masses[tid] - m) > _NUM_TOL:
                raise ValueError(
                    f"试管 {tid} 在 {mass_src[tid]} 中质量为 {masses[tid]:g}g, "
                    f"在 {_label(plan)} 中为 {m:g}g, 同编号试管质量不一致, 拒绝规划。"
                )
            if tid not in masses:
                masses[tid] = m
                mass_src[tid] = _label(plan)
    return masses


# ---------------------------------------------------------------------------
# Held-Karp 精确动态规划
# ---------------------------------------------------------------------------
def _optimal_order(load: list[int], keep: list[list[int]],
                   ids: list[int]) -> tuple[tuple[int, ...], int, int]:
    """返回 (最优方案ID序列, 最小总操作数, 达到最小值的序列数)。

    dp[mask][j] 记录覆盖 mask 且以 j 结尾的前缀的 (最小成本, 同成本下 ID
    序列字典序最小者, 达到最小成本的序列数); 转移成本只依赖相邻两方案。
    """
    k = len(ids)
    size = 1 << k
    inf = float("inf")
    dp_cost = [[inf] * k for _ in range(size)]
    dp_seq: list[list[Any]] = [[None] * k for _ in range(size)]
    dp_cnt = [[0] * k for _ in range(size)]
    for j in range(k):
        dp_cost[1 << j][j] = load[j]          # 空转子 → 首批: 仅装入
        dp_seq[1 << j][j] = (ids[j],)
        dp_cnt[1 << j][j] = 1
    for mask in range(size):
        for j in range(k):
            c0 = dp_cost[mask][j]
            if c0 == inf:
                continue
            s0 = dp_seq[mask][j]
            n0 = dp_cnt[mask][j]
            keep_j = keep[j]
            load_j = load[j]
            for t in range(k):
                if (mask >> t) & 1:
                    continue
                c = c0 + load_j + load[t] - 2 * keep_j[t]
                m2 = mask | (1 << t)
                s = s0 + (ids[t],)
                if c < dp_cost[m2][t]:
                    dp_cost[m2][t] = c
                    dp_seq[m2][t] = s
                    dp_cnt[m2][t] = n0
                elif c == dp_cost[m2][t]:
                    dp_cnt[m2][t] += n0
                    if s < dp_seq[m2][t]:
                        dp_seq[m2][t] = s
    full = size - 1
    best = min(range(k), key=lambda j: (dp_cost[full][j], dp_seq[full][j]))
    total = dp_cost[full][best]
    count = sum(dp_cnt[full][j] for j in range(k) if dp_cost[full][j] == total)
    return dp_seq[full][best], int(total), count


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def plan_changeover_sequence(plans: list[dict[str, Any]]) -> dict[str, Any]:
    """对 2~10 条已保存方案(GET /api/plan/{id} 返回结构)规划批次换装顺序。"""
    masses = validate_compatible(plans)
    by_id = {int(p["id"]): p for p in plans}
    ids = sorted(by_id)  # DP 内部顺序不影响结果(裁决按 ID 序列), 排序仅为确定性
    k = len(ids)

    placements = {pid: _placement_map(by_id[pid]) for pid in ids}
    # 防御: 摆放明细中的试管若不在请求清单, 用明细中的标称质量补充
    for pid in ids:
        for p in (by_id[pid].get("result") or {}).get("placements") or []:
            tid = int(p["tube_id"])
            if tid not in masses and p.get("mass") is not None:
                masses[tid] = float(p["mass"])

    load = [len(placements[pid]) for pid in ids]
    keep = [[0] * k for _ in range(k)]
    for i in range(k):
        pi = placements[ids[i]]
        for j in range(i + 1, k):
            pj = placements[ids[j]]
            shared = sum(1 for t, s in pi.items() if pj.get(t) == s)
            keep[i][j] = keep[j][i] = shared

    order, total, count = _optimal_order(load, keep, ids)

    # 逐批生成卸装清单(先卸后装), 并累计操作数
    steps: list[dict[str, Any]] = []
    cumulative = 0
    prev: dict[int, int] = {}
    for seq, pid in enumerate(order, start=1):
        target = placements[pid]
        unload = [
            {"tube_id": t, "from_slot": s, "mass": masses.get(t)}
            for t, s in prev.items() if target.get(t) != s
        ]
        load_list = [
            {"tube_id": t, "to_slot": s, "mass": masses.get(t)}
            for t, s in target.items() if prev.get(t) != s
        ]
        unload.sort(key=lambda x: (x["from_slot"], x["tube_id"]))
        load_list.sort(key=lambda x: (x["to_slot"], x["tube_id"]))
        ops = len(unload) + len(load_list)
        cumulative += ops
        steps.append({
            "seq": seq,
            "plan_id": pid,
            "plan_name": by_id[pid].get("name"),
            "unload": unload,
            "load": load_list,
            "unload_count": len(unload),
            "load_count": len(load_list),
            "operations": ops,
            "cumulative_operations": cumulative,
        })
        prev = target
    assert cumulative == total, "DP 总成本与逐批累计不一致"  # pragma: no cover

    return {
        "order": list(order),
        "plans": [
            {"id": pid, "name": by_id[pid].get("name"),
             "tube_count": len(placements[pid])}
            for pid in order
        ],
        "steps": steps,
        "total_operations": total,
        "optimal_sequence_count": count,
        "tie_broken": count > 1,
        "method": "Held-Karp 精确动态规划(状态数 2^k·k), 同成本按方案 ID 序列升序裁决",
        "readonly": True,
    }
