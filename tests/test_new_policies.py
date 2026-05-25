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


# =========================================================
# DeepACO 策略封装测试
# =========================================================

class TestDeepACOPolicyWrapper:

    def test_get_policy_name(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'n_ants': 30, 'n_iterations_train': 1, 'n_iterations_test': 5})
        assert w.get_policy_name() == 'deepaco'

    def test_default_params(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({})
        assert w.n_ants == 30
        assert w.n_iterations_train == 1
        assert w.n_iterations_test == 5

    def test_custom_params(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'n_ants': 50, 'n_iterations_train': 2, 'n_iterations_test': 10})
        assert w.n_ants == 50
        assert w.n_iterations_train == 2
        assert w.n_iterations_test == 10

    def test_validate_n_ants_zero_invalid(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'n_ants': 0})
        valid, _ = w.validate_config()
        assert not valid

    def test_validate_n_ants_too_large(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'n_ants': 201})
        valid, _ = w.validate_config()
        assert not valid

    def test_validate_n_iterations_train_zero_invalid(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'n_ants': 30, 'n_iterations_train': 0})
        valid, _ = w.validate_config()
        assert not valid

    def test_policy_params_includes_deepaco_keys(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'n_ants': 20, 'n_iterations_train': 1, 'n_iterations_test': 3})
        params = w.get_policy_params()
        assert params['n_ants'] == 20
        assert params['n_iterations_train'] == 1
        assert params['n_iterations_test'] == 3


# =========================================================
# DeepACO 注册 + 兼容性测试
# =========================================================

class TestDeepACORegistration:

    def test_in_policy_registry(self):
        from modules.policies import POLICY_REGISTRY
        assert 'deepaco' in POLICY_REGISTRY

    def test_in_policy_info(self):
        from modules.policies import POLICY_INFO
        assert 'deepaco' in POLICY_INFO
        assert POLICY_INFO['deepaco']['type'] == 'hybrid-aco'

    def test_get_policy_class_returns_wrapper(self):
        from modules.policies import get_policy_class
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        cls = get_policy_class('deepaco')
        assert cls is DeepACOPolicyWrapper


class TestDeepACOCompatibility:

    def test_compat_tsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'tsp') is True

    def test_compat_cvrp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'cvrp') is True

    def test_compat_mtsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'mtsp') is True

    def test_compat_op(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'op') is True

    def test_not_compat_pctsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'pctsp') is False

    def test_not_compat_pdp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'pdp') is False

    def test_not_compat_atsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'atsp') is False

    def test_algo_reinforce_ok(self):
        from modules.compatibility import is_policy_compatible_with_algorithm
        assert is_policy_compatible_with_algorithm('deepaco', 'reinforce') is True

    def test_algo_ppo_not_ok(self):
        from modules.compatibility import is_policy_compatible_with_algorithm
        assert is_policy_compatible_with_algorithm('deepaco', 'ppo') is False

    def test_algo_a2c_not_ok(self):
        from modules.compatibility import is_policy_compatible_with_algorithm
        assert is_policy_compatible_with_algorithm('deepaco', 'a2c') is False
