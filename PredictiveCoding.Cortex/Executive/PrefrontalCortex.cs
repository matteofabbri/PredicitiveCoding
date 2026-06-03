using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Gated;
using PredictiveCoding.Expanded.HierarchicalMoE;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// PrefrontalCortex  —  working memory + hierarchical task selection
//
// Models the executive role of PFC:
//   • a gated working-memory cell (GatedStateNetwork) maintains task-relevant
//     context across steps, writing in surprising information and holding the
//     rest — supporting multi-step reasoning;
//   • a hierarchical mixture of experts (HierarchicalMixtureOfExperts) selects a
//     task-set / response from the maintained context via energy competition.
//
// References: Miller & Cohen (2001) PFC theory; O'Reilly & Frank (2006) gated WM.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record PrefrontalCortexConfig
{
    /// <summary>Gated working-memory cell.</summary>
    public GatedStateNetworkConfig WorkingMemory { get; init; } = new();

    /// <summary>Hierarchical task-set experts.  ExpertConfig.LayerSizes[0] must
    /// equal the working-memory latent dim.</summary>
    public HierarchicalMoEConfig TaskExperts { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/pfc";

    public void Validate()
    {
        WorkingMemory.Validate();
        TaskExperts.Validate();
        int wmDim = WorkingMemory.NetworkConfig.LayerSizes[^1];
        if (TaskExperts.ExpertConfig.LayerSizes[0] != wmDim)
            throw new InvalidOperationException(
                $"TaskExperts.ExpertConfig.LayerSizes[0] ({TaskExperts.ExpertConfig.LayerSizes[0]}) must equal " +
                $"the working-memory latent dim ({wmDim}).");
    }
}

public sealed class PrefrontalCortex : IDisposable
{
    private readonly GatedStateNetwork             _wm;
    private readonly HierarchicalMixtureOfExperts  _experts;

    public PrefrontalCortex(PrefrontalCortexConfig config, Random rng)
    {
        config.Validate();
        _wm      = new GatedStateNetwork(config.WorkingMemory with { BasePath = Path.Combine(config.BasePath, "wm") }, new Random(rng.Next()));
        _experts = new HierarchicalMixtureOfExperts(config.TaskExperts with { BasePath = Path.Combine(config.BasePath, "experts") }, new Random(rng.Next()));
    }

    /// <summary>Updates working memory with a new observation; returns the WM state.</summary>
    public DenseTensor<float> Step(DenseTensor<float> input, int steps) => _wm.Step(input, steps);

    /// <summary>Updates WM, then trains the task experts on the maintained context.</summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int steps)
    {
        var context = _wm.Step(input, steps);
        _experts.Learn(context, label, steps);
    }

    /// <summary>Updates WM, then selects a task-set / response from the context.</summary>
    public int Decide(DenseTensor<float> input, int steps)
    {
        var context = _wm.Step(input, steps);
        return _experts.Predict(context, steps);
    }

    /// <summary>Clears working memory (start of a new trial / task episode).</summary>
    public void Reset() => _wm.Reset();

    public void FlushAll()
    {
        _wm.FlushStore();
        _experts.FlushAll();
    }

    public void Dispose()
    {
        _wm.Dispose();
        _experts.Dispose();
    }
}
