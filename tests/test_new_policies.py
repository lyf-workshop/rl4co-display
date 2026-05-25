"""新策略模型（MDAM、DeepACO）的单元测试"""
import pytest


# =========================================================
# MDAM 策略封装测试
# =========================================================

class TestMDAMPolicyWrapper:

    def test_get_policy_name(self):
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        w = MDAMPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3, 'num_heads': 8})
        assert w.get_policy_name() == 'mdam'

    def test_validate_config_valid(self):
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        w = MDAMPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3, 'num_heads': 8})
        valid, msg = w.validate_config()
        assert valid, msg

    def test_validate_config_embed_not_divisible(self):
        """embed_dim=100 不能被 num_heads=8 整除，BasePolicy 应拒绝"""
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        w = MDAMPolicyWrapper({'embed_dim': 100, 'num_encoder_layers': 3, 'num_heads': 8})
        valid, _ = w.validate_config()
        assert not valid

    def test_policy_params_returns_standard_keys(self):
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        w = MDAMPolicyWrapper({'embed_dim': 64, 'num_encoder_layers': 2, 'num_heads': 4})
        params = w.get_policy_params()
        assert 'embed_dim' in params
        assert params['embed_dim'] == 64


# =========================================================
# MDAM 注册 + 兼容性测试
# =========================================================

class TestMDAMRegistration:

    def test_in_policy_registry(self):
        from modules.policies import POLICY_REGISTRY
        assert 'mdam' in POLICY_REGISTRY

    def test_in_policy_info(self):
        from modules.policies import POLICY_INFO
        assert 'mdam' in POLICY_INFO
        assert POLICY_INFO['mdam']['type'] == 'multi-decoder-constructive'

    def test_get_policy_class_returns_wrapper(self):
        from modules.policies import get_policy_class
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        cls = get_policy_class('mdam')
        assert cls is MDAMPolicyWrapper


class TestMDAMCompatibility:

    def test_compat_tsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'tsp') is True

    def test_compat_cvrp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'cvrp') is True

    def test_compat_op(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'op') is True

    def test_compat_pctsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'pctsp') is True

    def test_not_compat_atsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'atsp') is False

    def test_not_compat_ffsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'ffsp') is False

    def test_algo_reinforce_ok(self):
        from modules.compatibility import is_policy_compatible_with_algorithm
        assert is_policy_compatible_with_algorithm('mdam', 'reinforce') is True

    def test_algo_ppo_not_ok(self):
        from modules.compatibility import is_policy_compatible_with_algorithm
        assert is_policy_compatible_with_algorithm('mdam', 'ppo') is False
