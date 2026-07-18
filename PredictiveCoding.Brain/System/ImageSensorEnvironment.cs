namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ImageSensorEnvironment  —  IEnvironment whose objects are real images.
//
// The real-sensor embodiment of the roadmap: the "object" is a real image (any
// camera frame or decoded photo) and the "sensor" is a small square patch that
// scans across it — visual scanning per P2/P9: a sensor moving over an object
// and reporting local features at each fixation. Dependency-free: images enter
// as float[,] luminance in [0, 1], indexed [y, x] (row, column); converting any
// RGB frame to that is a one-liner (e.g. 0.299·R + 0.587·G + 0.114·B).
//
// Sensor positions form a grid of ((W−patch)/stride+1) × ((H−patch)/stride+1)
// fixations. At each fixation Observe() reports:
//   Location — PoseEncoder.EncodeXY(col, row) over the grid extents (the
//              encoder is shared across objects, so all images of one
//              environment must share the same dimensions);
//   Feature  — a deterministic patch descriptor: the DOMINANT gradient
//              orientation as a category (central-difference gradients folded
//              to [0°, 180°); +1 reserved category for flat patches) plus the
//              dominant bin's energy share and the mean patch luminance as
//              scalar tails, bit-offset concatenated (the same pattern as
//              PoseEncoder.Encode). Same patch → identical SDR; same texture
//              orientation → high overlap; different orientation → low overlap.
//
// Motor command mapping (see MoveRight … MoveRandom constants):
//   0  MoveRight  — one grid step right (clamps at the border)
//   1  MoveDown   — one grid step down  (row +1; image rows grow downward)
//   2  MoveLeft   — one grid step left  (clamps at the border)
//   3  MoveUp     — one grid step up    (clamps at the border)
//   4  MoveRandom — jump to a uniformly random fixation
//
// Reference: Clay et al. (2024) P2 (sensor modules over real sensory data);
// Leadholm et al. (2025) P9 §3 — embodied sensorimotor loop.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Environment where objects are real images explored by a movable sensor patch.</summary>
public sealed class ImageSensorEnvironment : IEnvironment
{
    private readonly Dictionary<int, float[,]> _objects = [];
    private int _currentObjectId = -1;
    private int _col;
    private int _row;
    private readonly Random _rng;

    private readonly int _patchSize;
    private readonly int _stride;
    private readonly int _orientationBins;
    private readonly int _binBits;
    private readonly int _binActiveBits;
    private readonly ScalarEncoder _componentEnc;   // shared (0,1) encoder for the scalar tail blocks
    private readonly CategoryEncoder _dominantEnc;  // dominant orientation (+1 reserved "flat patch" category)

    private PoseEncoder? _locationEnc;              // built over the grid extents at first AddObject
    private int _imageWidth = -1;                   // shared image dimensions, fixed by the first AddObject
    private int _imageHeight = -1;
    private int _gridWidth;
    private int _gridHeight;

    // ── Location-encoder constants ───────────────────────────────────────────

    private const int PositionBits = 100;           // bits per positional axis (PoseEncoder default)
    private const int PositionActiveBits = 10;      // active bits per positional axis

    // ── Motor command constants ──────────────────────────────────────────────

    /// <summary>Motor command: move the sensor one grid step right (clamps at the border).</summary>
    public const int MoveRight = 0;

    /// <summary>Motor command: move the sensor one grid step down (clamps at the border).</summary>
    public const int MoveDown = 1;

    /// <summary>Motor command: move the sensor one grid step left (clamps at the border).</summary>
    public const int MoveLeft = 2;

    /// <summary>Motor command: move the sensor one grid step up (clamps at the border).</summary>
    public const int MoveUp = 3;

    /// <summary>Motor command: jump the sensor to a uniformly random fixation.</summary>
    public const int MoveRandom = 4;

    // ── IEnvironment ─────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public int ObjectCount => _objects.Count;

    /// <inheritdoc/>
    public int CurrentObjectId => _currentObjectId;

    /// <inheritdoc/>
    public int AvailableActions => 5;

    // ── Additional properties ────────────────────────────────────────────────

    /// <summary>Number of horizontal sensor positions: (imageWidth − patchSize)/stride + 1.
    /// 0 until the first object is registered.</summary>
    public int GridWidth => _gridWidth;

    /// <summary>Number of vertical sensor positions: (imageHeight − patchSize)/stride + 1.
    /// 0 until the first object is registered.</summary>
    public int GridHeight => _gridHeight;

    /// <summary>Current sensor position on the fixation grid (column, row).</summary>
    public (int Col, int Row) SensorPosition => (_col, _row);

    /// <summary>Size in bits of the Feature SDR returned by <see cref="Observe"/>:
    /// (orientationBins + 1) ScalarEncoder blocks of binBits each. Use it to size
    /// downstream column inputs.</summary>
    public int FeatureSize => _dominantEnc.Size + 2 * _binBits;

    /// <summary>Size in bits of the Location SDR returned by <see cref="Observe"/>
    /// (PoseEncoder.EncodeXY: two positional blocks). Use it to size downstream
    /// column inputs.</summary>
    public int LocationSize => 2 * PositionBits;

    // ── Constructor ──────────────────────────────────────────────────────────

    /// <param name="patchSize">Side of the square sensor patch, in pixels (≥ 4). Default 16.</param>
    /// <param name="stride">Grid step between neighbouring fixations, in pixels (≥ 1). Default 8.</param>
    /// <param name="orientationBins">Orientation-histogram bins over [0°, 180°) (≥ 2). Default 8.</param>
    /// <param name="binBits">Total bits per descriptor component (N). Default 32.</param>
    /// <param name="binActiveBits">Active bits per descriptor component (W). Default 6.</param>
    /// <param name="seed">RNG seed for <see cref="MoveRandom"/>. Default 42.</param>
    public ImageSensorEnvironment(
        int patchSize = 16, int stride = 8, int orientationBins = 8,
        int binBits = 32, int binActiveBits = 6, int seed = 42)
    {
        if (patchSize < 4)
            throw new ArgumentOutOfRangeException(nameof(patchSize), "Must be ≥ 4.");
        if (stride < 1)
            throw new ArgumentOutOfRangeException(nameof(stride), "Must be ≥ 1.");
        if (orientationBins < 2)
            throw new ArgumentOutOfRangeException(nameof(orientationBins), "Must be ≥ 2.");

        _patchSize = patchSize;
        _stride = stride;
        _orientationBins = orientationBins;
        _binBits = binBits;
        _binActiveBits = binActiveBits;
        _componentEnc = new ScalarEncoder(0, 1, binBits, binActiveBits);   // validates binBits/binActiveBits
        // The dominant-orientation block carries 3× the bits of a scalar tail so that
        // orientation identity dominates the code (disjoint categories ⇒ different
        // textures share few bits); the extra category encodes "flat, no gradient".
        _dominantEnc = new CategoryEncoder(orientationBins + 1, 3 * binActiveBits);
        _rng = new Random(seed);
    }

    // ── Object registration ──────────────────────────────────────────────────

    /// <summary>Registers <paramref name="luminance"/> as the image for
    /// <paramref name="objectId"/>. Overwrites any previously registered image
    /// for the same id. The image is luminance in [0, 1], indexed [y, x]
    /// (row, column); it must fit at least one sensor patch, and all images of
    /// one environment must share the same dimensions because the location
    /// encoder is shared across objects.</summary>
    public void AddObject(int objectId, float[,] luminance)
    {
        ArgumentNullException.ThrowIfNull(luminance);
        int height = luminance.GetLength(0);
        int width = luminance.GetLength(1);
        if (width < _patchSize || height < _patchSize)
            throw new ArgumentException(
                $"The image ({width}×{height}) must fit at least one {_patchSize}×{_patchSize} sensor patch.",
                nameof(luminance));

        if (_imageWidth < 0)
        {
            // First object fixes the shared dimensions and the location encoder.
            _imageWidth = width;
            _imageHeight = height;
            _gridWidth = (width - _patchSize) / _stride + 1;
            _gridHeight = (height - _patchSize) / _stride + 1;
            _locationEnc = new PoseEncoder(
                0, _gridWidth - 1, 0, _gridHeight - 1,
                PositionBits, PositionActiveBits);
        }
        else if (width != _imageWidth || height != _imageHeight)
        {
            throw new ArgumentException(
                $"All images of one environment must share the same dimensions " +
                $"({_imageWidth}×{_imageHeight} expected, {width}×{height} given): " +
                "the location encoder is shared across objects.",
                nameof(luminance));
        }

        _objects[objectId] = (float[,])luminance.Clone();
    }

    // ── IEnvironment ─────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public void PresentObject(int objectId)
    {
        if (!_objects.ContainsKey(objectId))
            throw new KeyNotFoundException(
                $"Object {objectId} is not registered in the environment.");
        _currentObjectId = objectId;
        (_col, _row) = (_gridWidth / 2, _gridHeight / 2);   // grid centre
    }

    /// <inheritdoc/>
    public (Sdr Feature, Sdr Location) Observe()
    {
        if (_currentObjectId < 0)
            throw new InvalidOperationException(
                "Call PresentObject before Observe.");
        var feature = DescribePatch(_objects[_currentObjectId]);
        var location = _locationEnc!.EncodeXY(_col, _row);
        return (feature, location);
    }

    /// <inheritdoc/>
    public void MoveTo(int motorCommand)
    {
        if (_currentObjectId < 0)
            throw new InvalidOperationException(
                "Call PresentObject before MoveTo.");
        if (motorCommand < 0 || motorCommand >= AvailableActions)
            throw new ArgumentOutOfRangeException(nameof(motorCommand),
                $"Motor command must be in [0, {AvailableActions}).");

        (_col, _row) = motorCommand switch
        {
            MoveRight => (Math.Min(_col + 1, _gridWidth - 1), _row),
            MoveDown  => (_col, Math.Min(_row + 1, _gridHeight - 1)),
            MoveLeft  => (Math.Max(_col - 1, 0), _row),
            MoveUp    => (_col, Math.Max(_row - 1, 0)),
            _         => (_rng.Next(_gridWidth), _rng.Next(_gridHeight)),   // MoveRandom
        };
    }

    /// <inheritdoc/>
    public void Reset() => (_col, _row) = (_gridWidth / 2, _gridHeight / 2);

    // ── Raw descriptor access ────────────────────────────────────────────────

    /// <summary>The raw descriptor at the current sensor position: the dominant-orientation
    /// category (== orientationBins for a flat patch), its energy share and the mean luminance.</summary>
    public (int Category, float Share, float Luminance) CurrentDescriptor()
    {
        if (_currentObjectId < 0)
            throw new InvalidOperationException(
                "Call PresentObject before CurrentDescriptor.");
        return ComputeDescriptor(_objects[_currentObjectId]);
    }

    // ── Patch descriptor ─────────────────────────────────────────────────────

    // Deterministic texture descriptor of the patch under the current fixation.
    // Three bit-offset blocks (the concatenation pattern of PoseEncoder.Encode):
    //   1. dominant gradient orientation as a CATEGORY (disjoint buckets, 3× the
    //      bits of a scalar tail, +1 reserved category for flat patches) — the
    //      identity component: patches with different dominant orientations share
    //      almost no bits. Encoding every bin's share instead would let the many
    //      zero-energy bins dominate the code and give unrelated textures ~80%
    //      baseline overlap — too close for downstream SDR thresholds.
    //   2. the dominant bin's energy share (how "striped" the patch is);
    //   3. mean patch luminance.
    private Sdr DescribePatch(float[,] image)
    {
        var (category, share, meanLuminance) = ComputeDescriptor(image);

        // 4. Concatenate category + share + luminance with bit offsets, as
        //    PoseEncoder.Encode does.
        var catSdr = _dominantEnc.Encode(category);
        var shareSdr = _componentEnc.Encode(share);
        var lumSdr = _componentEnc.Encode(meanLuminance);

        var bits = new int[catSdr.Cardinality + shareSdr.Cardinality + lumSdr.Cardinality];
        int idx = 0;
        foreach (int b in catSdr.ActiveBits) bits[idx++] = b;
        foreach (int b in shareSdr.ActiveBits) bits[idx++] = b + _dominantEnc.Size;
        foreach (int b in lumSdr.ActiveBits) bits[idx++] = b + _dominantEnc.Size + _binBits;
        return new Sdr(FeatureSize, bits);
    }

    // The raw (category, share, luminance) triple behind DescribePatch — shared
    // by CurrentDescriptor so recorders can log the un-encoded descriptor.
    private (int Category, float Share, float Luminance) ComputeDescriptor(float[,] image)
    {
        int px = _col * _stride;   // patch origin, pixel coordinates
        int py = _row * _stride;

        // 1. Mean luminance over the full patch.
        float luminanceSum = 0f;
        for (int y = 0; y < _patchSize; y++)
            for (int x = 0; x < _patchSize; x++)
                luminanceSum += image[py + y, px + x];

        // 2. Orientation-energy histogram over interior pixels: central-difference
        //    gradients; orientation folded to [0°, 180°); magnitude accumulated.
        var energy = new float[_orientationBins];
        for (int y = 1; y < _patchSize - 1; y++)
        {
            for (int x = 1; x < _patchSize - 1; x++)
            {
                float gx = (image[py + y, px + x + 1] - image[py + y, px + x - 1]) * 0.5f;
                float gy = (image[py + y + 1, px + x] - image[py + y - 1, px + x]) * 0.5f;
                float m = MathF.Sqrt(gx * gx + gy * gy);
                if (m <= 0f) continue;

                float theta = MathF.Atan2(gy, gx);                       // (−π, π]
                if (theta < 0f) theta += MathF.PI;                       // fold to [0, π]
                int bin = (int)(theta / MathF.PI * _orientationBins);
                if (bin >= _orientationBins) bin -= _orientationBins;    // 180° ≡ 0°
                energy[bin] += m;
            }
        }

        // 3. Dominant orientation and its energy share. A flat patch (no gradient
        //    anywhere) takes the reserved category and share 0.
        float total = 0f;
        int dominant = 0;
        for (int i = 0; i < _orientationBins; i++)
        {
            total += energy[i];
            if (energy[i] > energy[dominant]) dominant = i;
        }
        int category = total > 0f ? dominant : _orientationBins;   // last bucket = flat
        float share = total > 0f ? energy[dominant] / total : 0f;
        float meanLuminance = luminanceSum / (_patchSize * _patchSize);

        return (category, share, meanLuminance);
    }
}
