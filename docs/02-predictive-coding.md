# 2. Predictive coding — the theory

Predictive coding (PC) frames perception and learning as **free-energy
minimisation** (Rao & Ballard 1999; Friston 2005; Bogacz 2017). It is the
mathematical core every other project builds on.

## The model

A hierarchical network has `L` layers. Each layer `l` holds a latent state
vector `μ_l` and generates a **top-down prediction** of the layer below:

```
pred_{l-1} = f(W_l · μ_l + b_l)
```

where `f` is a non-linearity (Tanh by default) and `W_l`, `b_l` are the
generative weights/biases. The **prediction error** at layer `l-1` is

```
ε_{l-1} = Π_{l-1} · (μ_{l-1} − pred_{l-1})
```

`Π` is a (scalar or diagonal) precision — the inverse variance, i.e. how much to
trust that error. The whole network minimises the **free energy**

```
F = ½ Σ_l ‖ε_l‖²
```

## Two nested updates

**Inference** (fast, per example) — settle the hidden states by gradient descent
on `F`, holding weights fixed:

```
μ_l ← μ_l − η_s · ( ε_l − W_lᵀ · δ_{l-1} + λ_act · sign(μ_l) )
δ_{l-1} = f'(pre_{l-1}) ⊙ ε_{l-1}
```

The input layer `μ_0` is **clamped** to the datum; during supervised learning the
top layer is clamped to the label.

**Learning** (slow) — after inference settles, update weights with a **local
Hebbian** rule (no backprop):

```
ΔW_l = η_w · δ_{l-1} ⊗ μ_lᵀ
Δb_l = η_w · δ_{l-1}
```

Every term is available locally at a layer — the product of its own
prediction error and the activity of the layer above. This is the property the
whole codebase exploits.

## Classification as energy minimisation

There is no softmax head. To classify, clamp the input and try each candidate
label, keeping the one that yields the lowest settled free energy:

```
c* = argmin_c  F( x, one-hot(c) )
```

This `argmin_c F` principle reappears throughout: the Mixture-of-Experts gates on
free energy, the cortex's readout uses it, and the LLM's energy-free path is a
deliberate departure (see [doc 6](06-llm.md)).

## Variants implemented in the core

| Idea | Reference | Where |
|------|-----------|-------|
| Hierarchical PC | Rao & Ballard 1999 | `Network` |
| Free-energy principle | Friston 2005 | `Layer.LocalEnergy` |
| Diagonal precision, L2 decay | Huang & Rao 2011 | `UseDiagonalPrecision`, `WeightDecay` |
| Feedforward Gist warm-start | Haider et al. 2024 | `UseFeedforwardGistInit` |
| L1 activity sparsity | Keller & Mrsic-Flogel 2018 | `ActivityRegularization` |
| Temporal continuity | Friston 2008 | `TemporalNetwork` |

## Why it matters here

Because learning is local and energy-based, the same engine generalises cleanly to
**modular hierarchies**, **temporal cascades**, **attractor memories**,
**active inference**, and **mixtures of experts** — all of which appear in the
[Expanded](04-expanded.md) and [Cortex](05-cortex.md) projects without ever
introducing global backpropagation.

## References
- Rao, R. & Ballard, D. (1999). *Predictive coding in the visual cortex.* Nat. Neurosci.
- Friston, K. (2005). *A theory of cortical responses.* Phil. Trans. R. Soc. B.
- Bogacz, R. (2017). *A tutorial on the free-energy framework…* J. Math. Psychol.
- Whittington, J. & Bogacz, R. (2017). *An approximation of the error backpropagation…* Neural Comput.
