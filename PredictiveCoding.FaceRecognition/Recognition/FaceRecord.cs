namespace PredictiveCoding.FaceRecognition.Recognition;

/// <summary>
/// A registered face identity stored in a <see cref="FaceDatabase"/>.
/// Each record maps a <see cref="Name"/> (human-readable label) to an
/// L2-normalised embedding vector and optional source metadata.
/// </summary>
public sealed class FaceRecord
{
    /// <summary>Unique identifier (UUID, auto-generated on registration).</summary>
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..12];

    /// <summary>Human-readable identity label (e.g. "Alice").</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>L2-normalised embedding vector (dimension depends on the embedder).</summary>
    public float[] Embedding { get; set; } = [];

    /// <summary>Path to the source image, if available.</summary>
    public string? ImagePath { get; set; }

    /// <summary>Detector confidence at registration time (1.0 if unknown / no detector).</summary>
    public float Quality { get; set; } = 1.0f;

    /// <summary>UTC timestamp of registration.</summary>
    public DateTime AddedAt { get; set; } = DateTime.UtcNow;
}
