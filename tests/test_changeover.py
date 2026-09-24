"""批次换装顺序规划测试: python -m unittest discover -s tests"""
import copy
import itertools
import unittest

from app.changeover import plan_changeover_sequence


def make_plan(pid, masses, placement, radius=100.0, n_slots=12, disabled=()):
    """构造 GET /api/plan/{id} 返回风格的方案字典。

    masses: {试管号: 质量}; placement: {试管号: 槽位}
    """
    slots = [{"index": i, "angle": i * 30.0, "enabled": i not in disabled}
             for i in range(n_slots)]
    tubes = [{"id": t, "mass": m} for t, m in sorted(masses.items())]
    placements = [
        {"tube_id": t, "slot": s, "mass": masses[t], "locked": False,
         "angle": slots[s]["angle"], "previous_slot": None, "moved": False}
        for t, s in sorted(placement.items(), key=lambda kv: kv[1])
    ]
    return {
        "id": pid,
        "name": f"方案{pid}",
        "created_at": "2026-09-24T00:00:00+00:00",
        "parent_id": None,
        "request": {"slot_count": n_slots, "slots": slots, "tubes": tubes,
                    "radius_mm": radius},
        "result": {"placements": placements},
    }


def _placement_of(plan):
    return {p["tube_id"]: p["slot"] for p in plan["result"]["placements"]}


def brute_force(plans):
    """暴力枚举全部排列, 返回 (最小成本, 该成本下 ID 升序最小序列, 最优序列数)。"""
    ids = [p["id"] for p in plans]
    pl = {p["id"]: _placement_of(p) for p in plans}
    best, best_seq, cnt = None, None, 0
    for perm in itertools.permutations(ids):
        cost = len(pl[perm[0]])
        for a, b in zip(perm, perm[1:]):
            keep = sum(1 for t, s in pl[a].items() if pl[b].get(t) == s)
            cost += len(pl[a]) + len(pl[b]) - 2 * keep
        if best is None or cost < best:
            best, best_seq, cnt = cost, perm, 1
        elif cost == best:
            cnt += 1
            best_seq = min(best_seq, perm)
    return best, list(best_seq), cnt


class TestChangeoverCost(unittest.TestCase):
    def test_cost_model_and_steps(self):
        # 管1 同槽保留; 管2 卸下, 管3 装入 -> 切换成本 2
        p1 = make_plan(1, {1: 50, 2: 50}, {1: 0, 2: 6})
        p2 = make_plan(2, {1: 50, 3: 40}, {1: 0, 3: 3})
        r = plan_changeover_sequence([p1, p2])
        # 两个方向成本相同(2+2), 按 ID 升序裁决为 [1, 2]
        self.assertEqual(r["order"], [1, 2])
        self.assertEqual(r["total_operations"], 4)
        self.assertEqual(r["optimal_sequence_count"], 2)
        self.assertTrue(r["tie_broken"])
        s1, s2 = r["steps"]
        # 首批: 空转子起点, 仅装入
        self.assertEqual(s1["unload"], [])
        self.assertEqual(s1["load_count"], 2)
        self.assertEqual(s1["operations"], 2)
        self.assertEqual(s1["cumulative_operations"], 2)
        # 第二批: 卸管2(槽6), 装管3(槽3)
        self.assertEqual([(u["tube_id"], u["from_slot"]) for u in s2["unload"]],
                         [(2, 6)])
        self.assertEqual([(u["tube_id"], u["to_slot"]) for u in s2["load"]],
                         [(3, 3)])
        self.assertEqual(s2["operations"], 2)
        self.assertEqual(s2["cumulative_operations"], 4)
        # 清单带质量
        self.assertEqual(s2["load"][0]["mass"], 40.0)

    def test_keep_in_place_reduces_cost(self):
        # A 与 B 完全同槽(keep=2), C 与两者均无共同 -> C 应排在端点
        a = make_plan(1, {1: 50, 2: 50}, {1: 0, 2: 6})
        b = make_plan(2, {1: 50, 2: 50, 3: 40}, {1: 0, 2: 6, 3: 3})
        c = make_plan(3, {9: 70}, {9: 9})
        r = plan_changeover_sequence([c, b, a])
        best, seq, cnt = brute_force([a, b, c])
        self.assertEqual(r["total_operations"], best)
        self.assertEqual(r["order"], seq)
        self.assertEqual(r["optimal_sequence_count"], cnt)
        # C 在端点且 A↔B 相邻: 装C(1) + C→A全换(3) + B增装管3(1) = 5
        self.assertEqual(best, 1 + 3 + 1)
        self.assertEqual(r["steps"][-1]["cumulative_operations"], best)

    def test_exact_vs_brute_six_plans(self):
        # 6 方案(720 排列)与暴力枚举对照: 成本、序列、最优解数完全一致
        plans = [
            make_plan(11, {1: 50, 2: 48, 3: 52}, {1: 0, 2: 4, 3: 8}),
            make_plan(12, {1: 50, 4: 45}, {1: 0, 4: 6}),
            make_plan(13, {2: 48, 3: 52, 5: 30}, {2: 4, 3: 10, 5: 2}),
            make_plan(14, {6: 60}, {6: 1}),
            make_plan(15, {1: 50, 6: 60, 7: 25}, {1: 0, 6: 1, 7: 7}),
            make_plan(16, {3: 52, 8: 33}, {3: 8, 8: 5}),
        ]
        r = plan_changeover_sequence(plans)
        best, seq, cnt = brute_force(plans)
        self.assertEqual(r["total_operations"], best)
        self.assertEqual(r["order"], seq)
        self.assertEqual(r["optimal_sequence_count"], cnt)
        # 逐批累计与总数一致, 且每批 卸+装 = 本批操作数
        self.assertEqual(sum(s["operations"] for s in r["steps"]), best)
        for s in r["steps"]:
            self.assertEqual(s["unload_count"] + s["load_count"], s["operations"])
            self.assertEqual(len(s["unload"]), s["unload_count"])
            self.assertEqual(len(s["load"]), s["load_count"])
        # 顺序是所选 ID 的一个排列
        self.assertEqual(sorted(r["order"]), [11, 12, 13, 14, 15, 16])

    def test_tie_break_ascending_ids(self):
        # 互不保留(任意顺序成本相同) -> 返回 ID 升序, 与输入顺序无关
        p7 = make_plan(7, {1: 50}, {1: 0})
        p3 = make_plan(3, {2: 50}, {2: 3})
        p9 = make_plan(9, {3: 50}, {3: 6})
        r = plan_changeover_sequence([p7, p3, p9])
        self.assertEqual(r["order"], [3, 7, 9])
        self.assertEqual(r["total_operations"], 1 + 2 + 2)
        self.assertEqual(r["optimal_sequence_count"], 6)
        self.assertTrue(r["tie_broken"])

    def test_ten_plans_run(self):
        # 上限 10 条: DP 正常运行, 反序同成本(成本模型反转对称)
        plans = [make_plan(i + 1, {i + 1: 40 + i}, {i + 1: i})
                 for i in range(10)]
        r = plan_changeover_sequence(plans)
        self.assertEqual(sorted(r["order"]), list(range(1, 11)))
        self.assertEqual(r["total_operations"], 1 + 2 * 9)
        self.assertEqual(r["steps"][-1]["cumulative_operations"],
                         r["total_operations"])

    def test_readonly_input_untouched(self):
        p1 = make_plan(1, {1: 50, 2: 50}, {1: 0, 2: 6})
        p2 = make_plan(2, {1: 50, 3: 40}, {1: 0, 3: 3})
        snapshot = copy.deepcopy([p1, p2])
        plan_changeover_sequence([p1, p2])
        self.assertEqual([p1, p2], snapshot)


class TestChangeoverValidation(unittest.TestCase):
    def test_reject_count(self):
        p1 = make_plan(1, {1: 50}, {1: 0})
        with self.assertRaisesRegex(ValueError, "2~10"):
            plan_changeover_sequence([p1])
        plans = [make_plan(i + 1, {1: 50}, {1: 0}) for i in range(11)]
        with self.assertRaisesRegex(ValueError, "2~10"):
            plan_changeover_sequence(plans)

    def test_reject_duplicate_ids(self):
        p1 = make_plan(1, {1: 50}, {1: 0})
        p2 = make_plan(2, {1: 50}, {1: 3})
        dup = dict(p1)  # 不同内容但 ID 相同
        with self.assertRaisesRegex(ValueError, "重复"):
            plan_changeover_sequence([p1, dup, p2])

    def test_reject_radius_mismatch(self):
        p1 = make_plan(1, {1: 50}, {1: 0}, radius=100.0)
        p2 = make_plan(2, {1: 50}, {1: 0}, radius=120.0)
        with self.assertRaisesRegex(ValueError, "半径"):
            plan_changeover_sequence([p1, p2])

    def test_reject_slot_angle_mismatch(self):
        p1 = make_plan(1, {1: 50}, {1: 0})
        p2 = make_plan(2, {1: 50}, {1: 0})
        p2["request"]["slots"][3]["angle"] = 91.0
        with self.assertRaisesRegex(ValueError, "角度"):
            plan_changeover_sequence([p1, p2])

    def test_reject_slot_enabled_mismatch(self):
        p1 = make_plan(1, {1: 50}, {1: 0})
        p2 = make_plan(2, {1: 50}, {1: 0}, disabled=(5,))
        with self.assertRaisesRegex(ValueError, "启用状态"):
            plan_changeover_sequence([p1, p2])

    def test_reject_slot_set_mismatch(self):
        p1 = make_plan(1, {1: 50}, {1: 0}, n_slots=12)
        p2 = make_plan(2, {1: 50}, {1: 0}, n_slots=10)
        with self.assertRaisesRegex(ValueError, "槽位"):
            plan_changeover_sequence([p1, p2])

    def test_reject_same_tube_different_mass(self):
        p1 = make_plan(1, {1: 50, 2: 48}, {1: 0, 2: 6})
        p2 = make_plan(2, {1: 50, 3: 40}, {1: 0, 3: 3})
        p3 = make_plan(3, {2: 49}, {2: 4})  # 管2 质量与 p1 不一致
        with self.assertRaisesRegex(ValueError, "试管 2.*质量"):
            plan_changeover_sequence([p1, p2, p3])

    def test_accept_same_geometry_different_tube_sets(self):
        # 试管清单可不同, 只要同编号质量一致
        p1 = make_plan(1, {1: 50, 2: 48}, {1: 0, 2: 6})
        p2 = make_plan(2, {1: 50, 3: 40}, {1: 0, 3: 3})
        r = plan_changeover_sequence([p1, p2])
        self.assertEqual(r["total_operations"], 4)


if __name__ == "__main__":
    unittest.main()
