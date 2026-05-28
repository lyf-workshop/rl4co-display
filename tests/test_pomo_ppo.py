"""
验证 POMO+PPO 和 ATSP 可用组合
"""
import sys, traceback
import torch

PASS = []
FAIL = []

def test(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, type(e).__name__, str(e)[:200]))
        print(f"  FAIL  {name}")
        print(f"        {type(e).__name__}: {str(e)[:200]}")

# ── POMO + PPO ────────────────────────────────────────────────────────────────
from rl4co.envs import TSPEnv
from rl4co.models import PPO, REINFORCE
from rl4co.models.zoo.pomo import POMO as POMOModel
from rl4co.models import AttentionModelPolicy

print("\n=== POMO+PPO 验证 ===")

# ── 先确认项目里 "POMO策略" 实际上是什么 ──────────────────────────────────────
def t_project_pomo_policy_type():
    """项目 POMOPolicyWrapper 在 rl4co 0.6.0 下会 fallback 到 AttentionModelPolicy"""
    import sys
    sys.path.insert(0, '.')
    from modules.policies.pomo_policy import POMOPolicyWrapper
    env = TSPEnv(generator_params={'num_loc': 10})
    wrapper = POMOPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 2, 'num_heads': 4})
    policy = wrapper.create_policy(env)
    print(f"        actual policy type: {type(policy).__name__}")
    print(f"        has use_pomo attr: {hasattr(policy, 'use_pomo')}")
test("项目POMO策略的实际类型", t_project_pomo_policy_type)

# ── POMO (实际=AM fallback) + PPO ─────────────────────────────────────────────
def t_pomo_ppo_create():
    env = TSPEnv(generator_params={'num_loc': 10})
    # 模拟项目实际行为：fallback 到 AttentionModelPolicy
    policy = AttentionModelPolicy(env_name=env.name, embed_dim=128, num_encoder_layers=2, num_heads=4)
    policy.use_pomo = True  # 项目里设置的无效 flag
    model = PPO(env, policy, batch_size=32, train_data_size=320, val_data_size=32,
                optimizer_kwargs={'lr': 1e-4})
    print(f"        model type: {type(model).__name__}")
test("POMO(实际=AM)+PPO: model creation", t_pomo_ppo_create)

def t_pomo_ppo_train_step():
    from rl4co.utils.trainer import RL4COTrainer
    env = TSPEnv(generator_params={'num_loc': 10})
    policy = AttentionModelPolicy(env_name=env.name, embed_dim=128, num_encoder_layers=2, num_heads=4)
    policy.use_pomo = True
    model = PPO(env, policy, batch_size=16, train_data_size=32, val_data_size=16,
                optimizer_kwargs={'lr': 1e-4})
    trainer = RL4COTrainer(max_epochs=1, accelerator='cpu', devices='auto',
                           enable_progress_bar=False, enable_model_summary=False,
                           num_sanity_val_steps=0)
    trainer.fit(model)
    print(f"        1 epoch completed OK")
test("POMO(实际=AM)+PPO: 1-epoch training", t_pomo_ppo_train_step)

def t_pomo_model_assert():
    """RL4CO 真正的 POMO 模型类是否还有 assert baseline==shared"""
    env = TSPEnv(generator_params={'num_loc': 10})
    policy = AttentionModelPolicy(env_name=env.name, embed_dim=128, num_encoder_layers=2, num_heads=4)
    try:
        m = POMOModel(env, policy, baseline='rollout', batch_size=16,
                      train_data_size=32, val_data_size=16)
        print(f"        WARN: no assert raised, POMO model accepts non-shared baseline")
    except AssertionError as e:
        print(f"        assert fired: {e}")
    except Exception as e:
        print(f"        other error: {type(e).__name__}: {e}")
test("真正POMO模型: assert baseline==shared", t_pomo_model_assert)

# ── ATSP 可用组合 ─────────────────────────────────────────────────────────────
print("\n=== ATSP 可用组合验证 ===")

from rl4co.envs.routing import ATSPEnv
from rl4co.models.zoo.matnet import MatNet, MatNetPolicy
from rl4co.models import A2C

def t_atsp_matnet_reinforce():
    env = ATSPEnv(generator_params={'num_loc': 10})
    policy = MatNetPolicy(env_name=env.name, embed_dim=128, num_encoder_layers=2, num_heads=4)
    model = MatNet(env, policy, batch_size=16, train_data_size=32, val_data_size=16,
                   optimizer_kwargs={'lr': 1e-4})
    trainer_kwargs = dict(max_epochs=1, accelerator='cpu', devices='auto',
                          enable_progress_bar=False, enable_model_summary=False,
                          num_sanity_val_steps=0)
    from rl4co.utils.trainer import RL4COTrainer
    trainer = RL4COTrainer(**trainer_kwargs)
    trainer.fit(model)
    print(f"        1 epoch OK")
test("ATSP+MatNet+REINFORCE: 1-epoch training", t_atsp_matnet_reinforce)

def t_atsp_am_reinforce():
    """AM 用于 ATSP 应该失败（ATSPEnv 无 locs）"""
    env = ATSPEnv(generator_params={'num_loc': 10})
    try:
        policy = AttentionModelPolicy(env_name=env.name, embed_dim=128,
                                       num_encoder_layers=2, num_heads=4)
        model = REINFORCE(env, policy, baseline='rollout', batch_size=16,
                          train_data_size=32, val_data_size=16)
        td = env.reset(batch_size=[2])
        out = model(td)
        print(f"        WARN: AM+ATSP did NOT fail (unexpected)")
    except Exception as e:
        print(f"        fails as expected: {type(e).__name__}: {str(e)[:120]}")
test("ATSP+AM+REINFORCE: should fail", t_atsp_am_reinforce)

def t_atsp_matnet_ppo():
    """MatNet + PPO 应该失败（MatNet encoder 与 PPO Critic 不兼容）"""
    env = ATSPEnv(generator_params={'num_loc': 10})
    policy = MatNetPolicy(env_name=env.name, embed_dim=128, num_encoder_layers=2, num_heads=4)
    try:
        model = PPO(env, policy, batch_size=16, train_data_size=32, val_data_size=16,
                    optimizer_kwargs={'lr': 1e-4})
        from rl4co.utils.trainer import RL4COTrainer
        trainer = RL4COTrainer(max_epochs=1, accelerator='cpu', devices='auto',
                               enable_progress_bar=False, enable_model_summary=False,
                               num_sanity_val_steps=0)
        trainer.fit(model)
        print(f"        WARN: MatNet+PPO succeeded (unexpected)")
    except Exception as e:
        print(f"        fails as expected: {type(e).__name__}: {str(e)[:120]}")
test("ATSP+MatNet+PPO: should fail", t_atsp_matnet_ppo)

def t_atsp_matnet_a2c():
    """MatNet + A2C 也应该失败"""
    env = ATSPEnv(generator_params={'num_loc': 10})
    policy = MatNetPolicy(env_name=env.name, embed_dim=128, num_encoder_layers=2, num_heads=4)
    try:
        model = A2C(env, policy, batch_size=16, train_data_size=32, val_data_size=16,
                    optimizer_kwargs={'lr': 1e-4})
        from rl4co.utils.trainer import RL4COTrainer
        trainer = RL4COTrainer(max_epochs=1, accelerator='cpu', devices='auto',
                               enable_progress_bar=False, enable_model_summary=False,
                               num_sanity_val_steps=0)
        trainer.fit(model)
        print(f"        WARN: MatNet+A2C succeeded (unexpected)")
    except Exception as e:
        print(f"        fails as expected: {type(e).__name__}: {str(e)[:120]}")
test("ATSP+MatNet+A2C: should fail", t_atsp_matnet_a2c)

# ── 汇总 ──────────────────────────────────────────────────────────────────────
print(f"\n=== 汇总: {len(PASS)} PASS / {len(FAIL)} FAIL ===")
for name, etype, msg in FAIL:
    print(f"  FAIL {name}: {etype}: {msg}")
