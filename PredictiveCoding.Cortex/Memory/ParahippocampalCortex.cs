using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// ParahippocampalCortex  —  scene / context gist (the "place area")
//
// The parahippocampal place area encodes the gist of a scene — the spatial layout
// and context of "where you are" — complementing the entorhinal grid code for metric
// position.  Implemented as a VisualCortex (used purely as a scene-gist encoder, no
// internal classifier) feeding a context PCN that recognises the place/context from
// the gist latent.
//
// References: Epstein & Kanwisher (1998) parahippocampal place area;
//             Hassabis & Maguire (2007) scene construction.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record ParahippocampalCortexConfig
{
    /// <summary>Scene encoder (a VisualCortex; its own Classifier is forced to null).</summary>
    public VisualCortexConfig Scene { get; init; } = new();

    /// <summary>Context / place readout.  LayerSizes[0] must equal the scene IT latent dim.</summary>
    public NetworkConfig Context { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/parahippocampal";

    public void Validate()
    {
        Scene.Validate();
        if (Context.LayerSizes[0] != Scene.ItLatentDim)
            throw new InvalidOperationException(
                $"Context.LayerSizes[0] ({Context.LayerSizes[0]}) must equal the scene IT latent dim ({Scene.ItLatentDim}).");
    }
}

public sealed class ParahippocampalCortex : IDisposable
{
    private readonly VisualCortex _scene;
    private readonly Network      _context;

    public int SceneLatentDim => _scene.ItLatentDim;
    public int ContextCount   => _context.Configuration.LayerSizes[^1];

    public ParahippocampalCortex(ParahippocampalCortexConfig config, Random rng)
    {
        config.Validate();
        _scene   = new VisualCortex(config.Scene with { Classifier = null, BasePath = Path.Combine(config.BasePath, "scene") }, new Random(rng.Next()));
        _context = new Network(config.Context with { Path = Path.Combine(config.BasePath, "context") }, new Random(rng.Next()));
    }

    /// <summary>Encodes a scene image into its gist latent.</summary>
    public DenseTensor<float> EncodeScene(DenseTensor<float> image, int steps) => _scene.Encode(image, steps);

    /// <summary>Learns the scene → context/place association (refines the gist, then trains the readout).</summary>
    public void Learn(DenseTensor<float> image, DenseTensor<float> contextLabel, int steps)
    {
        _scene.Learn(image, steps);
        _context.Learn(_scene.Encode(image, steps), contextLabel, steps);
    }

    /// <summary>Recognises the place / context of a scene.</summary>
    public int RecognizePlace(DenseTensor<float> image, int steps)
        => _context.Predict(_scene.Encode(image, steps), steps);

    public void FlushAll() { _scene.FlushAll(); _context.FlushStore(); }
    public void Dispose()  { _scene.Dispose();  _context.Dispose(); }
}
