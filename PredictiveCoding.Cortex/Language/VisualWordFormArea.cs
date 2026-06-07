using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// VisualWordFormArea (VWFA)  —  the "letterbox": orthography → pre-lexical code
//
// A reproducible patch of the left ventral occipitotemporal cortex that becomes tuned
// to written word forms with literacy. It is the visual gateway to language: it maps
// the visual appearance of a written word to an abstract, location- and font-invariant
// orthographic code that feeds the language network (`WernickeArea`). A PCN that
// learns visual-latent → orthographic-code, bridging `VisualCortex` and `Language`.
//
// References: Dehaene & Cohen (2011) the VWFA / "letterbox area";
//             McCandliss, Cohen & Dehaene (2003) the visual word form area.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record VisualWordFormAreaConfig
{
    /// <summary>Orthography PCN: visual latent → orthographic code.
    /// LayerSizes[^1] must equal <see cref="CodeDim"/>.</summary>
    public NetworkConfig Orthography { get; init; } = new();

    /// <summary>Dimensionality of the orthographic / pre-lexical code.</summary>
    public int CodeDim { get; init; } = 16;

    public string BasePath { get; init; } = "./cortex/vwfa";

    public void Validate()
    {
        if (CodeDim < 1)
            throw new InvalidOperationException("CodeDim must be >= 1.");
        if (Orthography.LayerSizes[^1] != CodeDim)
            throw new InvalidOperationException(
                $"Orthography.LayerSizes[^1] ({Orthography.LayerSizes[^1]}) must equal CodeDim ({CodeDim}).");
    }
}

public sealed class VisualWordFormArea : IDisposable
{
    private readonly Network _net;
    private readonly VisualWordFormAreaConfig _config;

    public int VisualDim => _net.Configuration.LayerSizes[0];
    public int CodeDim => _config.CodeDim;

    public VisualWordFormArea(VisualWordFormAreaConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        _net = new Network(config.Orthography with { Path = config.BasePath }, rng);
    }

    /// <summary>Maps a visual (word-image) latent to its abstract orthographic code,
    /// ready to feed the language network.</summary>
    public DenseTensor<float> Recognize(DenseTensor<float> visualLatent, int steps)
        => _net.InferGetTop(visualLatent, steps);

    /// <summary>Learns the visual-latent → orthographic-code association.</summary>
    public void Learn(DenseTensor<float> visualLatent, DenseTensor<float> orthographicCode, int steps)
    {
        if (orthographicCode.Length != _config.CodeDim)
            throw new ArgumentException(
                $"orthographicCode has {orthographicCode.Length} elements; expected CodeDim ({_config.CodeDim}).");
        _net.Learn(visualLatent, orthographicCode, steps);
    }

    public void FlushAll() => _net.FlushStore();
    public void Dispose() => _net.Dispose();
}
