"""
test_fix3_skip_optional_validity.py
------------------------------------
Verifies Fix 3: skip_optional_subprocess is now valid independently for
low-risk cases, without requiring relax_rules_for_low_risk to be called first.

Tests:
  1. skip_optional is valid when risk is low (no _skip_optional flag required)
  2. skip_optional is invalid when risk is high
  3. skip_optional can be used before relax_rules (no circular dependency)
  4. Applying skip_optional sets _skip_optional flag (for downstream logic)
  5. relax_rules still sets _skip_optional flag (backward compat)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from kpi_actions import (
    MANAGEMENT_ACTIONS,
    get_management_mask,
    apply_management_action,
    RISK_HIGH_THRESH,
)


class TestSkipOptionalValidity:

    def test_skip_optional_valid_for_low_risk_without_flag(self):
        """skip_optional should be valid for low-risk cases WITHOUT _skip_optional flag."""
        # Low risk: delay_norm + rework_norm < RISK_HIGH_THRESH
        kpi = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state = {"_risk_high": False}  # No _skip_optional flag set

        mask = get_management_mask(kpi, state)
        skip_idx = 7  # skip_optional_subprocess
        assert mask[skip_idx], (
            "skip_optional should be valid for low-risk cases even without _skip_optional flag. "
            "Fix 3 not applied correctly."
        )

    def test_skip_optional_invalid_for_high_risk(self):
        """skip_optional should be invalid when risk is high."""
        # High risk: delay_norm + rework_norm > RISK_HIGH_THRESH
        kpi = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state = {"_risk_high": False}

        mask = get_management_mask(kpi, state)
        skip_idx = 7
        assert not mask[skip_idx], (
            "skip_optional should be invalid for high-risk cases."
        )

    def test_skip_optional_invalid_when_risk_high_flag_set(self):
        """skip_optional should be invalid when _risk_high flag is True."""
        kpi = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state = {"_risk_high": True}  # Escalation set this flag

        mask = get_management_mask(kpi, state)
        skip_idx = 7
        assert not mask[skip_idx], (
            "skip_optional should be invalid when _risk_high flag is set."
        )

    def test_skip_optional_usable_before_relax_rules(self):
        """skip_optional can be used independently, before relax_rules is called."""
        kpi = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state = {"_risk_high": False}
        twin = None  # Not needed for skip_optional

        # Apply skip_optional directly (no relax_rules first)
        reward = apply_management_action(7, state, twin, kpi)
        assert reward == 0.0, "skip_optional should apply successfully."
        assert state.get("_skip_optional") is True, (
            "Applying skip_optional should set _skip_optional flag."
        )

    def test_skip_optional_sets_flag_when_applied(self):
        """Applying skip_optional should set _skip_optional flag in state."""
        kpi = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state = {}
        twin = None

        apply_management_action(7, state, twin, kpi)
        assert "_skip_optional" in state, "_skip_optional flag not set."
        assert state["_skip_optional"] is True, "_skip_optional should be True."

    def test_relax_rules_still_sets_skip_optional_flag(self):
        """relax_rules_for_low_risk should still set _skip_optional (backward compat)."""
        kpi = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state = {"_risk_high": False, "_objection_active": False, "_suspension_active": False}
        twin = None

        relax_idx = 11  # relax_rules_for_low_risk
        apply_management_action(relax_idx, state, twin, kpi)
        assert state.get("_skip_optional") is True, (
            "relax_rules should still set _skip_optional flag for backward compatibility."
        )

    def test_skip_optional_reduces_delay(self):
        """Applying skip_optional should reduce _delay_norm."""
        kpi = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state = {"_delay_norm": 1.5}
        twin = None

        apply_management_action(7, state, twin, kpi)
        assert state["_delay_norm"] < 1.5, (
            "skip_optional should reduce _delay_norm."
        )
        expected = max(0.0, 1.5 - 0.2)
        assert abs(state["_delay_norm"] - expected) < 1e-9, (
            f"Expected delay_norm={expected}, got {state['_delay_norm']}."
        )

    def test_no_circular_dependency(self):
        """
        Before Fix 3, skip_optional required _skip_optional=True (set by relax_rules),
        creating a circular dependency. After Fix 3, skip_optional is independent.
        """
        kpi = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state_no_flag = {"_risk_high": False}
        state_with_flag = {"_risk_high": False, "_skip_optional": True}

        mask_no_flag = get_management_mask(kpi, state_no_flag)
        mask_with_flag = get_management_mask(kpi, state_with_flag)

        skip_idx = 7
        # Both should be valid (no dependency on flag)
        assert mask_no_flag[skip_idx], (
            "skip_optional should be valid without _skip_optional flag (Fix 3)."
        )
        assert mask_with_flag[skip_idx], (
            "skip_optional should still be valid with _skip_optional flag."
        )


if __name__ == "__main__":
    suite = TestSkipOptionalValidity()
    tests = [m for m in dir(suite) if m.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            getattr(suite, t)()
            print(f"  PASS  {t}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
