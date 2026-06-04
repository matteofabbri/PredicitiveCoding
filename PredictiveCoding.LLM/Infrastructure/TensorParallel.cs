using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Infrastructure;

// ─────────────────────────────────────────────────────────────────────────────
// TensorParallel  —  sharded linear layers across a device mesh (Megatron-LM style).
//
//   • Column-parallel: the output dimension is split across devices; each device
//     computes a slice of y from the full x; the slices are gathered.
//   • Row-parallel: the input dimension is split; each device holds a column-slab
//     of W and the matching slice of x, computes a partial y, then the partials
//     are all-reduced (summed).
//
// "Devices" here are CPU workers (a DeviceMesh over threads); a real multi-GPU
// backend would implement IComputeDevice with cross-device collectives. The math
// is identical to single-device W·x + b.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>A compute device that runs a unit of work (CPU worker, or a GPU).</summary>
public interface IComputeDevice
{
    int Id { get; }
    void Execute(Action work);
}

/// <summary>A CPU "device" backed by a thread-pool task.</summary>
public sealed class CpuDevice(int id) : IComputeDevice
{
    public int Id { get; } = id;
    public void Execute(Action work) => work();
}

/// <summary>A set of devices that runs sharded work in parallel and synchronises.</summary>
public sealed class DeviceMesh
{
    public IReadOnlyList<IComputeDevice> Devices { get; }
    public int Count => Devices.Count;

    public DeviceMesh(int deviceCount)
    {
        if (deviceCount < 1) throw new ArgumentOutOfRangeException(nameof(deviceCount));
        var d = new IComputeDevice[deviceCount];
        for (int i = 0; i < deviceCount; i++) d[i] = new CpuDevice(i);
        Devices = d;
    }

    /// <summary>Runs <paramref name="shardWork"/> once per device, in parallel, and waits.</summary>
    public void Dispatch(Action<int> shardWork)
    {
        var tasks = new Task[Count];
        for (int s = 0; s < Count; s++)
        {
            int shard = s;
            var dev   = Devices[shard];
            tasks[shard] = Task.Run(() => dev.Execute(() => shardWork(shard)));
        }
        Task.WaitAll(tasks);
    }

    /// <summary>All-reduce (sum) of per-device partial vectors into a single vector.</summary>
    public static float[] AllReduceSum(float[][] partials)
    {
        var outp = new float[partials[0].Length];
        foreach (var p in partials)
            for (int i = 0; i < outp.Length; i++) outp[i] += p[i];
        return outp;
    }
}

/// <summary>A linear layer W·x + b whose work is sharded across a device mesh.</summary>
public sealed class ShardedLinear
{
    private readonly DenseTensor<float> _w;   // [out, in]
    private readonly float[]            _b;   // [out]
    private readonly DeviceMesh         _mesh;
    private readonly int                _out, _in;

    public ShardedLinear(DenseTensor<float> w, float[] bias, DeviceMesh mesh)
    {
        if (w.Rank != 2) throw new ArgumentException("weight must be [out, in].", nameof(w));
        _w = w; _b = bias; _mesh = mesh;
        _out = w.Dimensions[0];
        _in  = w.Dimensions[1];
        if (bias.Length != _out) throw new ArgumentException("bias length must equal output dim.");
    }

    /// <summary>Column-parallel: split the output rows across devices, then gather.</summary>
    public DenseTensor<float> ForwardColumnParallel(DenseTensor<float> x)
    {
        var y  = new DenseTensor<float>(_out);
        var wb = _w.Buffer; var xb = x.Buffer; var yb = y.Buffer;

        _mesh.Dispatch(shard =>
        {
            var (lo, hi) = Range(_out, _mesh.Count, shard);
            for (int r = lo; r < hi; r++)
            {
                float s = _b[r];
                int baseIdx = r * _in;
                for (int c = 0; c < _in; c++) s += wb[baseIdx + c] * xb[c];
                yb[r] = s;                       // each device writes a disjoint slice
            }
        });
        return y;
    }

    /// <summary>Row-parallel: split the input dim across devices, then all-reduce the partials.</summary>
    public DenseTensor<float> ForwardRowParallel(DenseTensor<float> x)
    {
        var partials = new float[_mesh.Count][];
        var wb = _w.Buffer; var xb = x.Buffer;

        _mesh.Dispatch(shard =>
        {
            var (lo, hi) = Range(_in, _mesh.Count, shard);
            var part = new float[_out];
            for (int r = 0; r < _out; r++)
            {
                float s = 0f;
                int baseIdx = r * _in;
                for (int c = lo; c < hi; c++) s += wb[baseIdx + c] * xb[c];
                part[r] = s;
            }
            partials[shard] = part;
        });

        var reduced = DeviceMesh.AllReduceSum(partials);
        var y = new DenseTensor<float>(_out);
        for (int r = 0; r < _out; r++) y.Buffer[r] = reduced[r] + _b[r];
        return y;
    }

    // Contiguous shard [lo, hi) of a dimension of size total split into shards.
    private static (int lo, int hi) Range(int total, int shards, int shard)
    {
        int per = (total + shards - 1) / shards;
        int lo  = Math.Min(shard * per, total);
        int hi  = Math.Min(lo + per, total);
        return (lo, hi);
    }
}
