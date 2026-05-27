"""Diagnostic: print reward shape from DeepACO policy forward pass."""
import torch

def main():
    from rl4co.envs import TSPEnv
    from rl4co.models.zoo.deepaco import DeepACO, DeepACOPolicy

    env = TSPEnv(generator_params={"num_loc": 20})
    batch = env.reset(batch_size=[4])

    # Test 1: default policy (no n_ants dict)
    print("=== Test 1: default policy ===")
    policy_default = DeepACOPolicy(env_name="tsp")
    print(f"  policy.n_ants = {policy_default.n_ants}")
    out = policy_default(batch.clone(), env, phase="train")
    print(f"  out['reward'].shape = {out['reward'].shape}")
    print(f"  out['log_likelihood'].shape = {out['log_likelihood'].shape}")

    # Test 2: dict n_ants (our current approach)
    print("=== Test 2: dict n_ants ===")
    policy_dict = DeepACOPolicy(env_name="tsp", n_ants={'train': 10, 'val': 10, 'test': 10})
    print(f"  policy.n_ants = {policy_dict.n_ants}")
    out = policy_dict(batch.clone(), env, phase="train")
    print(f"  out['reward'].shape = {out['reward'].shape}")

    # Test 3: int n_ants
    print("=== Test 3: int n_ants ===")
    policy_int = DeepACOPolicy(env_name="tsp", n_ants=10)
    print(f"  policy.n_ants = {policy_int.n_ants}")
    out = policy_int(batch.clone(), env, phase="train")
    print(f"  out['reward'].shape = {out['reward'].shape}")

    # Test 4: DeepACO model with pre-built policy, train_with_local_search=False
    print("=== Test 4: DeepACO model with pre-built policy ===")
    policy4 = DeepACOPolicy(env_name="tsp", n_ants={'train': 10, 'val': 10, 'test': 10})
    model = DeepACO(env, policy4, train_with_local_search=False,
                    batch_size=4, train_data_size=16, val_data_size=8)
    td = env.reset(batch_size=[4])
    out4 = model.policy(td.clone(), env, phase="train")
    print(f"  out['reward'].shape = {out4['reward'].shape}")
    print(f"  model.train_with_local_search = {model.train_with_local_search}")
    print(f"  policy.train_with_local_search = {policy4.train_with_local_search}")


if __name__ == "__main__":
    main()
