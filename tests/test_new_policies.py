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
