# `rlg` — Architecture Sketch

A JAX-native reinforcement learning library aimed at policy-gradient and
value-based methods for games. The two design pressures are **speed** (everything
JIT-compilable, vectorizable, ideally `pmap`-friendly) and **composability**
(swap networks, losses, target-network strategies, buffers without rewriting
the rest).

This document defines contracts (abstract classes, protocols, dataclass
signatures) only. No implementations.

---

## 1. Design principles

1. **Functional core, thin object shell.** Every hot-path function — losses,
   updates, environment steps, advantage estimation — is a pure function of
   pytrees. Classes exist only to hold *static* configuration and to advertise
   contracts; they never own mutable state.
2. **State is explicit and pytree-shaped.** No hidden globals, no member
   `self._params`. Parameters, optimizer state, target params, RNG keys, buffer
   contents — all live in plain dataclasses (registered as pytrees) that flow
   through `jit`/`vmap`/`scan`.
3. **Static vs. traced separation.** Anything that affects compilation
   (network architecture, batch shapes, algorithm choice) is `static`; anything
   numeric is `traced`. Networks are Flax `linen.Module`s — architecture is
   static, params are traced.
4. **Composition over inheritance.** Higher-level pieces (an actor-critic agent,
   a PPO trainer) are *constructed from* lower-level pieces. Inheritance is
   used only to declare contracts (ABCs / `Protocol`s).
5. **No leaky coupling.** An `Actor` knows nothing about a `Critic` and vice
   versa. A loss knows nothing about how a target network is updated. A trainer
   knows nothing about whether the backbone is shared. Cross-component knowledge
   lives at the *composition site*, not in the components.
6. **Speed budget is a first-class concern.** Every abstraction is justified
   against "can this still `jit` end-to-end?" If it can't, it's wrong.

---

## 2. Layer overview

The library is organized into ten layers; each depends only on layers below it.

|Layer|Name|Purpose|
|-|-|-|
|0|`types`|Shared pytrees, type aliases, specs|
|1|`networks`|Flax `linen.Module`s: torsos, heads, composite nets|
|2|`policies`|Distribution wrappers turning network outputs into actions|
|3|`envs`|Vectorized JAX environment interface|
|4|`buffers`|Rollout and replay storage|
|5|`agents`|Agent state container + factory contracts|
|6|`losses`|Pure loss functions over (params, batch, key)|
|7|`optim`|Optax wrappers + target-network strategies|
|8|`learners`|One JIT-compiled gradient step|
|9|`trainers`|Outer loop: collect → store → learn → log|
|10|`logging`|Side-effecting metric sinks|

---

## 3. Directory layout

```
rlg/
  types.py              # Layer 0
  networks/
    base.py             # Torso, Head ABCs
    torsos.py           # MLP, ConvTorso, ResidualTorso (signatures only)
    heads.py            # CategoricalHead, GaussianHead, QHead, DistributionalQHead
    composite.py        # SeparateActorCritic, SharedActorCritic, QNetwork, DuelingQ
  policies/
    base.py             # Policy ABC
    stochastic.py       # CategoricalPolicy, GaussianPolicy, TanhSquashedGaussian
    deterministic.py    # DeterministicPolicy, EpsilonGreedyPolicy
  envs/
    base.py             # VecEnv ABC, EnvSpec
    wrappers.py         # FrameStack, RewardClip, ActionRepeat (signatures)
  buffers/
    base.py             # Buffer ABC
    rollout.py          # RolloutBuffer
    replay.py           # UniformReplay, PrioritizedReplay
    nstep.py            # NStepWrapper
  agents/
    state.py            # AgentState pytree
    base.py             # Agent ABC
  losses/
    base.py             # LossFn protocol
    pg.py               # reinforce_loss, a2c_loss, ppo_loss, vmpo_loss
    value.py            # mse_value_loss, huber_value_loss
    qlearning.py        # dqn_loss, double_dqn_loss, distributional_q_loss
    actor_critic.py     # sac_loss, td3_loss, ddpg_loss
  optim/
    optimizer.py        # Optimizer wrapper around optax
    target.py           # TargetStrategy ABC + None/Hard/Polyak
  learners/
    base.py             # Learner ABC, LearnerStep protocol
    standard.py         # SingleNetworkLearner, ActorCriticLearner
  trainers/
    base.py             # Trainer ABC
    on_policy.py        # OnPolicyTrainer (PPO / A2C)
    off_policy.py       # OffPolicyTrainer (DQN / SAC / TD3)
  advantage.py          # gae, n_step_return, retrace, v_trace
  logging/
    base.py             # Logger ABC
    sinks.py            # StdoutLogger, TensorboardLogger, WandbLogger
```

---

## 4. Layer 0 — Core types

```python
# types.py
from typing import Protocol, TypeVar, Generic
import jax
import jax.numpy as jnp
from flax.struct import dataclass

PRNGKey      = jax.Array
Params       = Any  # frozen pytree of jnp.ndarray (Flax FrozenDict)
OptState     = Any  # optax pytree
Observation  = Any  # pytree, shape = (batch, *obs_shape)
Action       = Any  # pytree
Reward       = jax.Array
Done         = jax.Array
Info         = dict[str, jax.Array]

@dataclass
class EnvStep:
    obs:      Observation
    action:   Action
    reward:   Reward
    done:     Done
    next_obs: Observation
    info:     Info

@dataclass
class TimeStep:
    """Single env transition emitted by VecEnv.step."""
    obs:      Observation
    reward:   Reward
    done:     Done
    info:     Info

@dataclass
class Spec:
    shape: tuple[int, ...]
    dtype: jnp.dtype

@dataclass
class EnvSpec:
    obs_spec:    Spec
    action_spec: Spec
    discrete:    bool
    n_actions:   int | None   # None for continuous
```

`flax.struct.dataclass` makes these registered pytrees so they pass through
`jit`/`vmap` unchanged.

---

## 5. Layer 1 — Networks

Networks are Flax `linen.Module`s. Architecture is **static** (lives on the
module instance), parameters are **traced** (live in the params pytree).
Two contracts: `Torso` (obs → features) and `Head` (features → output).

```python
# networks/base.py
import flax.linen as nn

class Torso(nn.Module):
    """Feature extractor. Subclasses define `__call__(obs) -> features`."""
    feature_dim: int
    def __call__(self, obs: Observation) -> jax.Array: ...

class Head(nn.Module):
    """Maps features to a task-specific output (logits, mean+std, value, ...)."""
    def __call__(self, features: jax.Array) -> Any: ...
```

Composite networks:

```python
# networks/composite.py
class SeparateActorCritic(nn.Module):
    """Actor and critic with their own torsos. No parameter sharing."""
    actor_torso:  Torso
    actor_head:   Head
    critic_torso: Torso
    critic_head:  Head

    def __call__(self, obs):
        return (
            self.actor_head(self.actor_torso(obs)),
            self.critic_head(self.critic_torso(obs)),
        )

class SharedActorCritic(nn.Module):
    """Single shared torso fed into actor and critic heads."""
    torso:       Torso
    actor_head:  Head
    critic_head: Head

    def __call__(self, obs):
        features = self.torso(obs)
        return self.actor_head(features), self.critic_head(features)

class QNetwork(nn.Module):
    torso: Torso
    head:  Head     # produces (batch, n_actions) Q-values
    def __call__(self, obs): return self.head(self.torso(obs))
```

**Key invariant:** the loss layer never imports `SeparateActorCritic` or
`SharedActorCritic`. It only depends on the *interface*: a callable
`apply_fn(params, obs) -> (dist_params, value)`. The composite network decides
how to satisfy that interface; the loss is oblivious to whether parameters are
shared. This is what makes "shared backbone vs separate" a one-line config
change with no downstream edits.

---

## 6. Layer 2 — Policies & distributions

A `Policy` adapts the raw network output into something an environment can
consume (an action) and what a loss needs (log-probs, entropy).

```python
# policies/base.py
class Policy(Protocol):
    def sample(self,   network_out: Any, key: PRNGKey) -> Action: ...
    def mode(self,     network_out: Any) -> Action: ...
    def log_prob(self, network_out: Any, action: Action) -> jax.Array: ...
    def entropy(self,  network_out: Any) -> jax.Array: ...
```

Concrete: `CategoricalPolicy`, `GaussianPolicy`, `TanhSquashedGaussian`,
`DeterministicPolicy`, `EpsilonGreedyPolicy(epsilon: ScheduleFn)`.

`Policy` is a stateless object — all state (network params, ε schedule progress)
is passed in. It is a strategy object, not a stateful agent.

---

## 7. Layer 3 — Environments

Vectorized, fully JIT-able. Compatible by design with Gymnax / Brax / JaxMARL.

```python
# envs/base.py
class VecEnv(Protocol):
    spec: EnvSpec
    num_envs: int

    def reset(self, key: PRNGKey) -> tuple[EnvState, Observation]: ...
    def step(self, state: EnvState, action: Action, key: PRNGKey
            ) -> tuple[EnvState, TimeStep]: ...
```

`EnvState` is opaque to the trainer — it is whatever pytree the env carries
between steps. No Python-side mutation, no callbacks. Wrappers
(`FrameStack`, `RewardClip`, `ActionRepeat`) implement the same protocol and
hold a wrapped env as a static attribute.

---

## 8. Layer 4 — Buffers

```python
# buffers/base.py
class Buffer(Protocol):
    """All methods are pure: state in, state out."""
    def init(self, dummy: EnvStep) -> BufferState: ...
    def add(self, state: BufferState, step: EnvStep) -> BufferState: ...
    def sample(self, state: BufferState, key: PRNGKey, batch_size: int
              ) -> tuple[BufferState, Batch]: ...
    @property
    def is_ready(self) -> Callable[[BufferState], bool]: ...
```

- `RolloutBuffer` — fixed-length contiguous storage; `sample` returns
  *the whole rollout* shaped `(T, num_envs, ...)`. Used for on-policy.
- `UniformReplay` — circular buffer; `sample` returns a uniform minibatch.
- `PrioritizedReplay` — adds priority pytree + `update_priorities` method.
- `NStepWrapper` — wraps any buffer, accumulates n-step returns at write time.

`BufferState` is a pytree — fixed-shape arrays preallocated at `init` so the
whole training step compiles to one XLA graph.

---

## 9. Layer 5 — Agent state

The single source of truth that flows through every step.

```python
# agents/state.py
@dataclass
class AgentState:
    params:        Params
    opt_state:     OptState
    target_params: Params | None       # None when no target net used
    buffer_state:  BufferState
    env_state:     EnvState
    rng:           PRNGKey
    step:          jax.Array            # int, scalar
```

`AgentState` is a pytree — `jax.tree.map`-able, JIT-friendly. **Every learner
and trainer step has signature `AgentState -> AgentState` (plus metrics)**.
This makes checkpoints trivial (one `eqx.tree_serialise_leaves` /
`flax.serialization.to_bytes` call) and `jax.lax.scan` over training steps
straightforward.

```python
# agents/base.py
class Agent(Protocol):
    """Bundles the static config needed to produce and consume an AgentState."""
    network:   nn.Module
    policy:    Policy
    optimizer: Optimizer
    target:    TargetStrategy
    loss_fn:   LossFn

    def init(self, key: PRNGKey, env: VecEnv) -> AgentState: ...
    def act(self, state: AgentState, obs: Observation, key: PRNGKey
           ) -> Action: ...
```

`Agent` is a tiny façade — its job is to wire the pieces and to define `init`
and `act`. All training logic lives in the `Learner`.

---

## 10. Layer 6 — Losses

The most important contract in the library.

```python
# losses/base.py
class LossFn(Protocol):
    def __call__(self,
                 params:        Params,
                 target_params: Params | None,
                 batch:         Batch,
                 key:           PRNGKey,
                 ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Return (scalar loss, aux metrics)."""
```

- The signature is **fixed** for every loss in the library, regardless of
  algorithm. PPO, SAC, DQN, distributional Q, IMPALA — all conform.
- `target_params` is `None` when no target net is in use; the loss handles
  that internally. The trainer never branches on "do we have targets?".
- `batch` is whatever the buffer emitted; the loss declares (via type alias)
  what fields it expects. Mismatches surface at construction, not at runtime.
- The loss does **one forward pass per network**. For shared-backbone
  actor-critic, that one pass returns `(dist_params, value)` together; the
  loss then reuses both. This is the perf-critical path.

Concrete loss factories return functions, not classes:

```python
def make_ppo_loss(network: nn.Module,
                  policy:  Policy,
                  clip_eps: float,
                  vf_coef:  float,
                  ent_coef: float,
                 ) -> LossFn: ...
```

Static config is captured by closure; the returned callable is what gets
`jax.grad`'d.

---

## 11. Layer 7 — Optimizers & target-network strategies

```python
# optim/optimizer.py
@dataclass
class Optimizer:
    """Thin wrapper around optax.GradientTransformation."""
    tx: optax.GradientTransformation
    def init(self, params: Params)        -> OptState: ...
    def update(self, grads, opt_state, params) -> tuple[Params, OptState]: ...
```

```python
# optim/target.py
class TargetStrategy(Protocol):
    """Decides how target_params relate to params."""
    def init(self, params: Params) -> Params | None: ...
    def update(self, params: Params, target_params: Params | None, step: int
              ) -> Params | None: ...

class NoTarget(TargetStrategy):       """init -> None; update -> None"""
class HardUpdate(TargetStrategy):     """copy every `period` steps"""    period: int
class PolyakUpdate(TargetStrategy):   """target = τ·params + (1-τ)·target"""  tau: float
```

This is the answer to the Polyak-vs-no-Polyak requirement: it's a
single field swap on the agent factory:

```python
agent = ActorCritic(... , target=PolyakUpdate(tau=0.005))     # SAC-style
agent = ActorCritic(... , target=NoTarget())                  # plain A2C
agent = QAgent(...      , target=HardUpdate(period=10_000))   # DQN
```

The loss receives `target_params` (possibly `None`); the learner runs
`target.update(...)` once per step. Nothing else changes.

---

## 12. Layer 8 — Learners (one JIT step)

```python
# learners/base.py
class Learner(Protocol):
    """One self-contained gradient step. JIT-compiled once at construction."""
    def init(self, key: PRNGKey, env: VecEnv) -> AgentState: ...
    def step(self, state: AgentState, batch: Batch
            ) -> tuple[AgentState, dict[str, jax.Array]]: ...
```

A standard learner's `step` does:

1. `grads, aux = jax.grad(loss_fn, has_aux=True)(state.params, state.target_params, batch, key)`
2. `new_params, new_opt = optimizer.update(grads, state.opt_state, state.params)`
3. `new_target = target_strategy.update(new_params, state.target_params, state.step)`
4. return `replace(state, params=new_params, opt_state=new_opt, target_params=new_target, step=state.step+1)`, `aux`

The learner is **algorithm-agnostic**; PPO, SAC, DQN all share the same
`SingleNetworkLearner`. Differences live entirely in the `LossFn` and the
`Buffer`.

For algorithms that maintain *two independent parameter pytrees* (SAC's actor
and twin critics, DDPG's actor and critic), provide
`MultiNetworkLearner` with a list of `(params, opt_state, target_params,
loss_fn)` tuples.

---

## 13. Layer 9 — Trainers (outer loop)

A trainer ties the agent to the environment and the buffer. Two flavors:

```python
# trainers/on_policy.py
class OnPolicyTrainer:
    agent:       Agent
    env:         VecEnv
    buffer:      RolloutBuffer
    rollout_len: int
    n_epochs:    int
    n_minibatches: int
    advantage_fn: Callable    # gae, v_trace, ...
    logger:      Logger

    def train(self, state: AgentState, n_iterations: int) -> AgentState: ...
```

The on-policy `train` loop, written as `lax.scan` over iterations:

```
for _ in range(n_iterations):
    state, rollout = collect_rollout(state, env, agent, rollout_len)
    rollout        = advantage_fn(rollout, state.params)
    for epoch in range(n_epochs):
        for mb in shuffle_minibatches(rollout, n_minibatches):
            state, metrics = learner.step(state, mb)
    logger.write(metrics)
```

Off-policy mirror:

```python
class OffPolicyTrainer:
    agent:           Agent
    env:             VecEnv
    buffer:          ReplayBuffer
    warmup_steps:    int
    update_every:    int
    updates_per_step: int
```

```
for step in range(n_steps):
    state, transition = env_step(state, agent)
    state.buffer_state = buffer.add(state.buffer_state, transition)
    if step > warmup_steps and step % update_every == 0:
        for _ in range(updates_per_step):
            batch        = buffer.sample(state.buffer_state, key, batch_size)
            state, metrics = learner.step(state, batch)
```

Both loops compile to one XLA graph (modulo Python-side iteration controlled
by `lax.scan` / `lax.fori_loop`).

---

## 14. Layer 10 — Logging

```python
# logging/base.py
class Logger(Protocol):
    def write(self, metrics: dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...
```

The only impure layer. Called from the Python side **outside** any JIT
boundary, on host arrays already pulled from device. Implementations:
`StdoutLogger`, `TensorboardLogger`, `WandbLogger`, `MultiLogger(*sinks)`.

---

## 15. Composition examples

### 15.1 PPO with shared backbone

```python
network = SharedActorCritic(
    torso       = MLPTorso(hidden=(256, 256)),
    actor_head  = CategoricalHead(n_actions=env.spec.n_actions),
    critic_head = ValueHead(),
)
agent = Agent(
    network   = network,
    policy    = CategoricalPolicy(),
    optimizer = Optimizer(optax.adam(3e-4)),
    target    = NoTarget(),
    loss_fn   = make_ppo_loss(network, CategoricalPolicy(),
                              clip_eps=0.2, vf_coef=0.5, ent_coef=0.01),
)
trainer = OnPolicyTrainer(agent, env, RolloutBuffer(T=128), ..., advantage_fn=gae(λ=0.95, γ=0.99))
```

### 15.2 PPO with separate networks (zero changes elsewhere)

```python
network = SeparateActorCritic(
    actor_torso  = MLPTorso(hidden=(256, 256)),  actor_head  = CategoricalHead(...),
    critic_torso = MLPTorso(hidden=(256, 256)),  critic_head = ValueHead(),
)
# Everything below is identical to 15.1
```

### 15.3 SAC with Polyak targets

```python
agent = ActorCriticAgent(
    actor_network  = ActorOnly(MLPTorso(...), TanhGaussianHead(...)),
    critic_network = TwinQNetwork(MLPTorso(...), QHead()),
    policy         = TanhSquashedGaussian(),
    optimizer      = Optimizer(optax.adam(3e-4)),
    target         = PolyakUpdate(tau=0.005),         # ← only line that changes from "no target"
    loss_fn        = make_sac_loss(...),
)
trainer = OffPolicyTrainer(agent, env, UniformReplay(capacity=1_000_000), ...)
```

### 15.4 Rainbow DQN

```python
network = DuelingQ(
    torso = ConvTorso(...),
    value_head     = ValueHead(),
    advantage_head = AdvantageHead(n_actions),
)
agent = QAgent(
    network   = network,
    policy    = EpsilonGreedyPolicy(eps=linear_schedule(1.0, 0.05, 1_000_000)),
    optimizer = Optimizer(optax.adam(6.25e-5)),
    target    = HardUpdate(period=8000),
    loss_fn   = make_distributional_q_loss(n_atoms=51, v_min=-10, v_max=10),
)
trainer = OffPolicyTrainer(agent, env, PrioritizedReplay(α=0.6, β=...), ...)
```

---

## 16. JAX considerations

- **Static vs traced.** Network architecture, buffer capacity, batch sizes,
  `n_epochs`, algorithm choice → static (Python ints / dataclass fields not
  marked traced). Params, observations, RNG keys, step count → traced.
- **`jit` boundary.** Place at `learner.step` and at the rollout collector.
  The trainer's outer loop is Python, but each iteration body is a single
  compiled call. For maximum perf, wrap the entire `n_iterations` loop in
  `lax.scan`.
- **`vmap`.** Use it for ensembles (twin Q-nets, network ensembles) at the
  network-construction site, not inside losses.
- **`pmap`.** All major state pytrees must be pmap-shardable. `AgentState`
  is designed so that `jax.device_put_sharded` works directly.
- **Donation.** Mark large pytrees (`params`, `buffer_state`) as donated in
  `jit` to avoid copies.
- **No Python control flow on traced values.** Branches based on `done`,
  `is_ready`, schedules → `lax.cond` / `lax.select`. Loops → `lax.scan` /
  `lax.fori_loop`.
- **RNG discipline.** Every function that needs randomness takes a `PRNGKey`
  argument. No global state. `jax.random.split` at every fan-out.

---

## 17. What's intentionally left out of the abstractions

These are deliberately *not* core abstractions, to keep the surface small:

- Multi-agent / self-play orchestration. Can be built on top by composing
  multiple `Agent`s sharing an env; doesn't need its own layer.
- Distributed actor-learner (IMPALA-style) topology. Use `pmap` plus an
  external orchestrator (e.g. `launchpad`). The library stays single-process
  conceptually; `pmap` is the only multi-device primitive.
- Hyperparameter sweeps. Out of scope; use `hydra` or `wandb sweeps` over
  the agent factory.
- Curricula and env wrappers beyond a small built-in set — users compose
  their own `VecEnv` implementations.
- Recurrent policies (LSTM/GRU/Transformer-XL) are supported by the existing
  abstractions (carry an additional `recurrent_state` field in `AgentState`,
  thread it through `act` and `loss_fn`) but are not pre-wired.
