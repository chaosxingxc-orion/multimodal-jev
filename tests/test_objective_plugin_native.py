"""Run only in the pinned training image: no native_surrogate replacement."""
import pytest

from tests.test_objective_plugin import (
    test_default_exact_loss_gradient_and_padding as check_default,
    test_default_dp_microbatch_gradient_matches_full_batch as check_dp,
    test_editing_only_extension_changes_gradient as check_edit,
)
from tests.test_objective_trace import (
    test_trace_roundtrip_reward_loss_and_gradient as check_trace,
    test_actual_custom_reward_is_recorded as check_custom_reward_trace,
)


def test_native_verl_default_loss_and_gradient():
    check_default(None)


def test_native_verl_dp_microbatch_gradient():
    check_dp(None)


@pytest.mark.parametrize('change',['reward','loss'])
def test_native_verl_editable_objective_gradient(tmp_path,change):
    check_edit(tmp_path,None,change)


def test_native_verl_trace_replay():
    check_trace(None)


def test_native_verl_custom_reward_trace(tmp_path):
    check_custom_reward_trace(None,tmp_path)
