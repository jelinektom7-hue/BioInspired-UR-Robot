from gymnasium.envs.registration import register

# register(
#     id="gymnasium_env/GridWorld-v0",
#     entry_point="gymnasium_env.envs:GridWorldEnv",
# )

register(
    id="gymnasium_env/WhipWorld-v0",
    entry_point="gymnasium_env.envs:WhipWorldEnv",
)