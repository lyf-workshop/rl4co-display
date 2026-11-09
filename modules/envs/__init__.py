"""
自定义环境模块
包含扩展和自定义的RL环境实现
"""

try:
    from .vrptw_env_wrapper import CVRPEnvWithTimeWindows
except ImportError:
    # The chip-routing core can run without the optional RL4CO dependency.
    CVRPEnvWithTimeWindows = None

from .chip_routing_env import (
    ChipRoutingGenerator,
    ChipRoutingInstance,
    ChipRoutingSolution,
    GridRoutingEnv,
)

__all__ = [
    'CVRPEnvWithTimeWindows',
    'ChipRoutingGenerator',
    'ChipRoutingInstance',
    'ChipRoutingSolution',
    'GridRoutingEnv',
]
