namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// DisplacementModule  —  relative-pose / displacement-cell encoding.
//
// Encodes continuous 2-D displacements (dx, dy) as SDRs by concatenating two
// ScalarEncoder outputs — one per axis. These SDRs represent the relative pose
// between object features and enable prediction of the next sensed location
// given a movement command.  The Apply / Invert / Between helpers operate in
// continuous space so callers can pipe results to a PoseEncoder for a full SDR.
//
// Reference: Hawkins et al. (2019) Fig. 4; Leadholm et al. (2025) Section 3.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class DisplacementModule
{
    private readonly ScalarEncoder _dxEnc;
    private readonly ScalarEncoder _dyEnc;

    /// <summary>Total size of the displacement SDR (2 × bitsPerAxis).</summary>
    public int SdrSize => _dxEnc.Size + _dyEnc.Size;

    /// <param name="maxAbsDisplacement">Maximum absolute displacement along either axis.</param>
    /// <param name="bitsPerAxis">Total SDR bits per axis (N). Default 60.</param>
    /// <param name="activePerAxis">Active bits per axis (W). Default 10.</param>
    public DisplacementModule(float maxAbsDisplacement, int bitsPerAxis = 60, int activePerAxis = 10)
    {
        if (maxAbsDisplacement <= 0f)
            throw new ArgumentOutOfRangeException(nameof(maxAbsDisplacement), "Must be positive.");
        _dxEnc = new ScalarEncoder(-maxAbsDisplacement, maxAbsDisplacement, bitsPerAxis, activePerAxis);
        _dyEnc = new ScalarEncoder(-maxAbsDisplacement, maxAbsDisplacement, bitsPerAxis, activePerAxis);
    }

    /// <summary>Encodes displacement (dx, dy) as a concatenated scalar SDR of size
    /// <see cref="SdrSize"/>. Nearby displacements have high overlap.</summary>
    public Sdr Encode(float dx, float dy)
    {
        var xSdr = _dxEnc.Encode(dx);
        var ySdr = _dyEnc.Encode(dy);
        int offset = _dxEnc.Size;

        var bits = new int[xSdr.Cardinality + ySdr.Cardinality];
        int idx = 0;
        foreach (int b in xSdr.ActiveBits) bits[idx++] = b;
        foreach (int b in ySdr.ActiveBits) bits[idx++] = b + offset;

        return new Sdr(SdrSize, bits);
    }

    /// <summary>Applies displacement (dx, dy) to a source position in continuous space,
    /// returning the target coordinates for subsequent SDR encoding.</summary>
    public (float x, float y) Apply(float sourceX, float sourceY, float dx, float dy)
        => (sourceX + dx, sourceY + dy);

    /// <summary>Returns the inverse displacement that undoes (dx, dy).</summary>
    public (float dx, float dy) Invert(float dx, float dy) => (-dx, -dy);

    /// <summary>Computes the displacement vector from (x1, y1) to (x2, y2).</summary>
    public (float dx, float dy) Between(float x1, float y1, float x2, float y2)
        => (x2 - x1, y2 - y1);
}
