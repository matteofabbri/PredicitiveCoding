namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// PoseEncoder3D  —  (x, y, z, yaw, pitch, roll) → SDR encoder.
//
// The SE(3) twin of PoseEncoder.  Combines three ScalarEncoders for Cartesian
// position with one CategoryEncoder per Euler angle for discretised orientation
// into a single concatenated SDR.  Nearby poses share many active bits (high
// overlap); distant positions or different orientations yield disjoint
// representations — exactly the property the cortical column needs for
// location-conditioned feature prediction.
//
// Encode is the main entry point.  EncodeXYZ and EncodeOrientation are exposed
// for cases where only one component is needed (e.g. path integration updates
// position without changing orientation).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class PoseEncoder3D
{
    private readonly ScalarEncoder _xEnc;
    private readonly ScalarEncoder _yEnc;
    private readonly ScalarEncoder _zEnc;
    private readonly CategoryEncoder _yawEnc;
    private readonly CategoryEncoder _pitchEnc;
    private readonly CategoryEncoder _rollEnc;
    private readonly int _orientationBuckets;

    /// <summary>Total SDR size: xN + yN + zN + 3 × orientationBuckets × orientationW.</summary>
    public int Size => _xEnc.Size + _yEnc.Size + _zEnc.Size
                     + _yawEnc.Size + _pitchEnc.Size + _rollEnc.Size;

    /// <param name="minX">Minimum x coordinate of the navigable space.</param>
    /// <param name="maxX">Maximum x coordinate.</param>
    /// <param name="minY">Minimum y coordinate.</param>
    /// <param name="maxY">Maximum y coordinate.</param>
    /// <param name="minZ">Minimum z coordinate.</param>
    /// <param name="maxZ">Maximum z coordinate.</param>
    /// <param name="positionBits">Total bits per positional axis (N). Default 100.</param>
    /// <param name="positionActiveBits">Active bits per axis (W). Default 10.</param>
    /// <param name="orientationBuckets">Discrete bins per Euler angle, e.g. 8 = 45° steps. Default 8.</param>
    /// <param name="orientationActiveBits">Active bits per orientation category. Default 5.</param>
    public PoseEncoder3D(
        float minX, float maxX, float minY, float maxY, float minZ, float maxZ,
        int positionBits = 100, int positionActiveBits = 10,
        int orientationBuckets = 8, int orientationActiveBits = 5)
    {
        if (maxX < minX) throw new ArgumentException("maxX must be ≥ minX.");
        if (maxY < minY) throw new ArgumentException("maxY must be ≥ minY.");
        if (maxZ < minZ) throw new ArgumentException("maxZ must be ≥ minZ.");
        if (orientationBuckets < 2)
            throw new ArgumentOutOfRangeException(nameof(orientationBuckets), "Must be ≥ 2.");

        _xEnc = new ScalarEncoder(minX, maxX, positionBits, positionActiveBits);
        _yEnc = new ScalarEncoder(minY, maxY, positionBits, positionActiveBits);
        _zEnc = new ScalarEncoder(minZ, maxZ, positionBits, positionActiveBits);
        _yawEnc = new CategoryEncoder(orientationBuckets, orientationActiveBits);
        _pitchEnc = new CategoryEncoder(orientationBuckets, orientationActiveBits);
        _rollEnc = new CategoryEncoder(orientationBuckets, orientationActiveBits);
        _orientationBuckets = orientationBuckets;
    }

    /// <summary>Encodes position (x, y, z) only (no orientation component).</summary>
    public Sdr EncodeXYZ(float x, float y, float z)
    {
        var xSdr = _xEnc.Encode(x);
        var ySdr = _yEnc.Encode(y);
        var zSdr = _zEnc.Encode(z);
        int xSize = _xEnc.Size;
        int xySize = xSize + _yEnc.Size;

        var bits = new int[xSdr.Cardinality + ySdr.Cardinality + zSdr.Cardinality];
        int idx = 0;
        foreach (int b in xSdr.ActiveBits) bits[idx++] = b;
        foreach (int b in ySdr.ActiveBits) bits[idx++] = b + xSize;
        foreach (int b in zSdr.ActiveBits) bits[idx++] = b + xySize;

        return new Sdr(xySize + _zEnc.Size, bits);
    }

    /// <summary>Encodes orientation (yaw, pitch, roll) as three concatenated
    /// categorical SDRs. Angles are in degrees; each is normalised to [0, 360)
    /// before bucketing.</summary>
    public Sdr EncodeOrientation(float yawDeg, float pitchDeg, float rollDeg)
    {
        var yawSdr = EncodeAngle(_yawEnc, yawDeg);
        var pitchSdr = EncodeAngle(_pitchEnc, pitchDeg);
        var rollSdr = EncodeAngle(_rollEnc, rollDeg);
        int yawSize = _yawEnc.Size;
        int yawPitchSize = yawSize + _pitchEnc.Size;

        var bits = new int[yawSdr.Cardinality + pitchSdr.Cardinality + rollSdr.Cardinality];
        int idx = 0;
        foreach (int b in yawSdr.ActiveBits) bits[idx++] = b;
        foreach (int b in pitchSdr.ActiveBits) bits[idx++] = b + yawSize;
        foreach (int b in rollSdr.ActiveBits) bits[idx++] = b + yawPitchSize;

        return new Sdr(yawPitchSize + _rollEnc.Size, bits);
    }

    /// <summary>Encodes the full pose (x, y, z, yaw, pitch, roll) as a single
    /// concatenated SDR of size <see cref="Size"/>. The x block occupies bits
    /// [0, xN), y occupies [xN, 2xN), z occupies [2xN, 3xN), and the three
    /// orientation blocks occupy [3xN, Size).</summary>
    public Sdr Encode(float x, float y, float z, float yawDeg, float pitchDeg, float rollDeg)
    {
        var xyzSdr = EncodeXYZ(x, y, z);
        var orientationSdr = EncodeOrientation(yawDeg, pitchDeg, rollDeg);
        int xyzSize = _xEnc.Size + _yEnc.Size + _zEnc.Size;

        var bits = new int[xyzSdr.Cardinality + orientationSdr.Cardinality];
        int idx = 0;
        foreach (int b in xyzSdr.ActiveBits) bits[idx++] = b;
        foreach (int b in orientationSdr.ActiveBits) bits[idx++] = b + xyzSize;

        return new Sdr(Size, bits);
    }

    /// <summary>Buckets one Euler angle exactly like PoseEncoder.EncodeTheta:
    /// normalise to [0, 360), then map onto the encoder's discrete bins.</summary>
    private Sdr EncodeAngle(CategoryEncoder encoder, float angleDeg)
    {
        float normalised = ((angleDeg % 360f) + 360f) % 360f;
        int bucket = (int)(normalised / 360f * _orientationBuckets) % _orientationBuckets;
        return encoder.Encode(bucket);
    }
}
