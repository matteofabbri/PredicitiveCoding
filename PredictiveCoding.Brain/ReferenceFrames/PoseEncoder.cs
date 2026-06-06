namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// PoseEncoder  —  (x, y, θ) → SDR encoder.
//
// Combines two ScalarEncoders for Cartesian position with a CategoryEncoder for
// discretised orientation into a single concatenated SDR.  Nearby poses share
// many active bits (high overlap); distant positions or different headings yield
// disjoint representations — exactly the property the cortical column needs for
// location-conditioned feature prediction.
//
// Encode is the main entry point.  EncodeXY and EncodeTheta are exposed for
// cases where only one component is needed (e.g. path integration updates
// position without changing heading).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class PoseEncoder
{
    private readonly ScalarEncoder _xEnc;
    private readonly ScalarEncoder _yEnc;
    private readonly CategoryEncoder _thetaEnc;
    private readonly int _orientationBuckets;

    /// <summary>Total SDR size: xN + yN + orientationBuckets × orientationW.</summary>
    public int Size => _xEnc.Size + _yEnc.Size + _thetaEnc.Size;

    /// <param name="minX">Minimum x coordinate of the navigable space.</param>
    /// <param name="maxX">Maximum x coordinate.</param>
    /// <param name="minY">Minimum y coordinate.</param>
    /// <param name="maxY">Maximum y coordinate.</param>
    /// <param name="positionBits">Total bits per positional axis (N). Default 100.</param>
    /// <param name="positionActiveBits">Active bits per axis (W). Default 10.</param>
    /// <param name="orientationBuckets">Discrete orientation bins, e.g. 8 = 45° steps. Default 8.</param>
    /// <param name="orientationActiveBits">Active bits per orientation category. Default 5.</param>
    public PoseEncoder(
        float minX, float maxX, float minY, float maxY,
        int positionBits = 100, int positionActiveBits = 10,
        int orientationBuckets = 8, int orientationActiveBits = 5)
    {
        if (maxX < minX) throw new ArgumentException("maxX must be ≥ minX.");
        if (maxY < minY) throw new ArgumentException("maxY must be ≥ minY.");
        if (orientationBuckets < 2)
            throw new ArgumentOutOfRangeException(nameof(orientationBuckets), "Must be ≥ 2.");

        _xEnc = new ScalarEncoder(minX, maxX, positionBits, positionActiveBits);
        _yEnc = new ScalarEncoder(minY, maxY, positionBits, positionActiveBits);
        _thetaEnc = new CategoryEncoder(orientationBuckets, orientationActiveBits);
        _orientationBuckets = orientationBuckets;
    }

    /// <summary>Encodes position (x, y) only (no orientation component).</summary>
    public Sdr EncodeXY(float x, float y)
    {
        var xSdr = _xEnc.Encode(x);
        var ySdr = _yEnc.Encode(y);
        int xSize = _xEnc.Size;

        var bits = new int[xSdr.Cardinality + ySdr.Cardinality];
        int idx = 0;
        foreach (int b in xSdr.ActiveBits) bits[idx++] = b;
        foreach (int b in ySdr.ActiveBits) bits[idx++] = b + xSize;

        return new Sdr(xSize + _yEnc.Size, bits);
    }

    /// <summary>Encodes orientation θ as a categorical SDR.
    /// θ is in degrees; any value is normalised to [0, 360) before bucketing.</summary>
    public Sdr EncodeTheta(float thetaDeg)
    {
        float normalised = ((thetaDeg % 360f) + 360f) % 360f;
        int bucket = (int)(normalised / 360f * _orientationBuckets) % _orientationBuckets;
        return _thetaEnc.Encode(bucket);
    }

    /// <summary>Encodes the full pose (x, y, θ) as a single concatenated SDR of size
    /// <see cref="Size"/>. The x block occupies bits [0, xN), y occupies [xN, 2xN),
    /// and θ occupies [2xN, Size).</summary>
    public Sdr Encode(float x, float y, float thetaDeg)
    {
        var xySdr = EncodeXY(x, y);
        var thetaSdr = EncodeTheta(thetaDeg);
        int xySize = _xEnc.Size + _yEnc.Size;

        var bits = new int[xySdr.Cardinality + thetaSdr.Cardinality];
        int idx = 0;
        foreach (int b in xySdr.ActiveBits) bits[idx++] = b;
        foreach (int b in thetaSdr.ActiveBits) bits[idx++] = b + xySize;

        return new Sdr(Size, bits);
    }
}
