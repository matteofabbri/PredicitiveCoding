using PredictiveCoding.Onnx;
using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx.Tests;

/// <summary>
/// The in-house PCNX model format and the save-time optimizer: byte-level
/// round-trips of every graph element, and the invariant that folding /
/// elision / dead-code removal leave Run results bit-identical. All offline
/// on synthetic graphs; the real downloaded models are covered by the
/// integration tests in the same file (skip silently when not on disk).
/// </summary>
public class OnnxPcnxFormatTests
{
    // ── Helpers ───────────────────────────────────────────────────────────────

    private static OnnxTensorData Floats(string name, long[] dims, float[] data)
        => new() { Name = name, DataType = OnnxDataType.Float, Dims = dims, Floats = data };

    private static OnnxTensorData Longs(string name, long[] dims, long[] data)
        => new() { Name = name, DataType = OnnxDataType.Int64, Dims = dims, Longs = data };

    private static OnnxNode Node(string op, string[] inputs, string[] outputs,
        params (string Name, OnnxAttribute Attr)[] attrs)
    {
        var node = new OnnxNode { Name = $"{op}_{outputs.FirstOrDefault() ?? ""}", OpType = op, Inputs = inputs, Outputs = outputs };
        foreach ((string name, OnnxAttribute attr) in attrs)
        {
            attr.Name = name;
            node.Attributes[name] = attr;
        }
        return node;
    }

    private static OnnxGraph RoundTrip(OnnxGraph graph)
    {
        using var ms = new MemoryStream();
        PcnxFormat.Write(ms, graph);
        return PcnxFormat.Read(ms.ToArray());
    }

    /// <summary>X (shape [2,2]) → Add(X, A) → Relu → Y, with A an initializer.
    /// Small but exercises inputs, weights, a chain and an output.</summary>
    private static OnnxGraph RunnableGraph()
    {
        var g = new OnnxGraph { Name = "runnable", OpsetVersion = 13 };
        g.Inputs.Add(new OnnxValueInfo { Name = "X", Shape = new[] { 2, 2 } });
        g.Outputs.Add(new OnnxValueInfo { Name = "Y", Shape = new[] { 2, 2 } });
        g.Initializers["A"] = Floats("A", [2, 2], [-10f, 0.5f, -0.5f, 10f]);
        g.Nodes.Add(Node("Add", ["X", "A"], ["sum"]));
        g.Nodes.Add(Node("Relu", ["sum"], ["Y"]));
        return g;
    }

    private static Tensor RunSingle(OnnxGraph g, float[] x)
        => Executor.Run(g, new() { ["X"] = new Tensor([2, 2], x) })["Y"];

    // ── Format round-trip ─────────────────────────────────────────────────────

    [Fact]
    public void RoundTrip_PreservesEveryGraphElement()
    {
        var g = new OnnxGraph { Name = "everything", OpsetVersion = 17 };
        g.Inputs.Add(new OnnxValueInfo { Name = "in", Shape = new[] { 1, 3, -1, -1 } });   // dynamic dims
        g.Outputs.Add(new OnnxValueInfo { Name = "out", Shape = [] });

        g.Initializers["w"] = Floats("w", [2, 3], [1f, 2f, 3f, 4f, 5f, 6f]);
        g.Initializers["idx"] = Longs("idx", [3], [0, 2, 1]);
        g.Initializers["scalar"] = Floats("scalar", [], [42f]);                     // rank-0
        g.Initializers["fp16"] = new OnnxTensorData { Name = "fp16", DataType = 10, Dims = [4] }; // no payload

        var node = Node("Fake", ["in", "w", ""], ["mid", "out"],
            ("f", new OnnxAttribute { Type = OnnxAttributeType.Float, F = 1.5f }),
            ("i", new OnnxAttribute { Type = OnnxAttributeType.Int, I = -7 }),
            ("s", new OnnxAttribute { Type = OnnxAttributeType.String, S = "half_pixel" }),
            ("floats", new OnnxAttribute { Type = OnnxAttributeType.Floats, Floats = [0.1f, 0.2f] }),
            ("ints", new OnnxAttribute { Type = OnnxAttributeType.Ints, Ints = [1, 2, 3] }),
            ("strings", new OnnxAttribute { Type = OnnxAttributeType.Strings, Strings = ["a", "b"] }),
            ("t", new OnnxAttribute { Type = OnnxAttributeType.Tensor, T = Longs("tv", [2], [9, 8]) }));
        g.Nodes.Add(node);

        OnnxGraph r = RoundTrip(g);

        Assert.Equal(g.Name, r.Name);
        Assert.Equal(g.OpsetVersion, r.OpsetVersion);
        Assert.Equal("in", r.Inputs.Single().Name);
        Assert.Equal(new[] { 1, 3, -1, -1 }, r.Inputs.Single().Shape);
        Assert.Equal("out", r.Outputs.Single().Name);
        Assert.Empty(r.Outputs.Single().Shape);

        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, r.Initializers["w"].Floats);
        Assert.Equal(new[] { 2L, 3L }, r.Initializers["w"].Dims);
        Assert.Equal(new[] { 0L, 2L, 1L }, r.Initializers["idx"].Longs);
        Assert.Equal(new[] { 42f }, r.Initializers["scalar"].Floats);
        Assert.Empty(r.Initializers["scalar"].Dims);
        Assert.Equal(10, r.Initializers["fp16"].DataType);
        Assert.Null(r.Initializers["fp16"].Floats);
        Assert.Null(r.Initializers["fp16"].Longs);

        OnnxNode rn = r.Nodes.Single();
        Assert.Equal("Fake", rn.OpType);
        Assert.Equal(new[] { "in", "w", "" }, rn.Inputs);
        Assert.Equal(new[] { "mid", "out" }, rn.Outputs);
        Assert.Equal(1.5f, rn.GetFloat("f"));
        Assert.Equal(-7, rn.GetInt("i"));
        Assert.Equal("half_pixel", rn.GetString("s"));
        Assert.Equal(new[] { 0.1f, 0.2f }, rn.Attributes["floats"].Floats);
        Assert.Equal(new[] { 1L, 2L, 3L }, rn.GetInts("ints"));
        Assert.Equal(new[] { "a", "b" }, rn.Attributes["strings"].Strings);
        Assert.Equal(new[] { 9L, 8L }, rn.Attributes["t"].T!.Longs);
    }

    [Fact]
    public void RoundTrip_RunsIdentically()
    {
        OnnxGraph g = RunnableGraph();
        float[] x = [1f, 2f, 3f, 4f];

        Tensor expected = RunSingle(g, x);
        Tensor actual = RunSingle(RoundTrip(g), x);

        Assert.Equal(expected.Shape, actual.Shape);
        Assert.Equal(expected.Data, actual.Data);
    }

    [Fact]
    public void Read_RejectsWrongMagic_AndSniffingIsStrict()
    {
        Assert.Throws<InvalidDataException>(() => PcnxFormat.Read("not a model"u8.ToArray()));
        Assert.False(PcnxFormat.LooksLikePcnx("ONNX"u8.ToArray()));
    }

    // ── OnnxModel Save / sniffing Load ────────────────────────────────────────

    [Fact]
    public void Save_ThenLoad_SniffsFormatAndRuns()
    {
        string path = Path.Combine(Path.GetTempPath(), $"pcnx_{Guid.NewGuid():N}.pcnx");
        try
        {
            using var ms = new MemoryStream();
            PcnxFormat.Write(ms, RunnableGraph());
            OnnxModel model = OnnxModel.Load(ms.ToArray());   // PCNX bytes → sniffed

            model.Save(path);                                  // optimized save
            OnnxModel reloaded = OnnxModel.Load(path);         // file → sniffed

            Assert.Equal(new[] { "X" }, reloaded.InputNames);
            Assert.Equal(new[] { "Y" }, reloaded.OutputNames);
            float[] x = [-1f, 2f, -3f, 4f];
            Assert.Equal(
                model.Run(new() { ["X"] = new Tensor([2, 2], x) })["Y"].Data,
                reloaded.Run(new() { ["X"] = new Tensor([2, 2], x) })["Y"].Data);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ── Optimizer ─────────────────────────────────────────────────────────────

    [Fact]
    public void Optimize_FoldsConstantExpressions_BitIdentically()
    {
        // A and B constant → Add(A,B) folds; Mul(X, folded) must remain.
        var g = new OnnxGraph { Name = "fold", OpsetVersion = 13 };
        g.Inputs.Add(new OnnxValueInfo { Name = "X", Shape = new[] { 2, 2 } });
        g.Outputs.Add(new OnnxValueInfo { Name = "Y", Shape = new[] { 2, 2 } });
        g.Initializers["A"] = Floats("A", [2, 2], [1e-3f, 2.7182818f, -3.1415927f, 4e6f]);
        g.Initializers["B"] = Floats("B", [2, 2], [0.1f, 0.2f, 0.3f, 0.4f]);
        g.Nodes.Add(Node("Add", ["A", "B"], ["C"]));
        g.Nodes.Add(Node("Mul", ["X", "C"], ["Y"]));

        OnnxGraph opt = GraphOptimizer.Optimize(g);

        Assert.Single(opt.Nodes);
        Assert.Equal("Mul", opt.Nodes[0].OpType);
        Assert.True(opt.Initializers.ContainsKey("C"));

        float[] x = [5f, 6f, 7f, 8f];
        Assert.Equal(
            Executor.Run(g, new() { ["X"] = new Tensor([2, 2], x) })["Y"].Data,
            Executor.Run(opt, new() { ["X"] = new Tensor([2, 2], x) })["Y"].Data);
    }

    [Fact]
    public void Optimize_InlinesConstantNodes()
    {
        // The executor has no "Constant" kernel: the raw graph cannot even run.
        // After optimization the node is an initializer and the graph runs.
        var g = new OnnxGraph { Name = "const", OpsetVersion = 13 };
        g.Inputs.Add(new OnnxValueInfo { Name = "X", Shape = new[] { 2, 2 } });
        g.Outputs.Add(new OnnxValueInfo { Name = "Y", Shape = new[] { 2, 2 } });
        g.Nodes.Add(Node("Constant", [], ["K"],
            ("value", new OnnxAttribute { Type = OnnxAttributeType.Tensor, T = Floats("kv", [2, 2], [2f, 2f, 2f, 2f]) })));
        g.Nodes.Add(Node("Mul", ["X", "K"], ["Y"]));

        OnnxGraph opt = GraphOptimizer.Optimize(g);

        Assert.Single(opt.Nodes);
        Tensor y = Executor.Run(opt, new() { ["X"] = new Tensor([2, 2], [1f, 2f, 3f, 4f]) })["Y"];
        Assert.Equal(new[] { 2f, 4f, 6f, 8f }, y.Data);
    }

    [Fact]
    public void Optimize_ElidesIdentityChains_ButKeepsGraphOutputProducers()
    {
        // X → Identity → t → Identity → u → Relu → mid → Identity → Y(output).
        // The two inner identities disappear; the one producing the declared
        // output name Y must stay (it is the caller's contract).
        var g = new OnnxGraph { Name = "identity", OpsetVersion = 13 };
        g.Inputs.Add(new OnnxValueInfo { Name = "X", Shape = new[] { 2, 2 } });
        g.Outputs.Add(new OnnxValueInfo { Name = "Y", Shape = new[] { 2, 2 } });
        g.Nodes.Add(Node("Identity", ["X"], ["t"]));
        g.Nodes.Add(Node("Identity", ["t"], ["u"]));
        g.Nodes.Add(Node("Relu", ["u"], ["mid"]));
        g.Nodes.Add(Node("Identity", ["mid"], ["Y"]));

        OnnxGraph opt = GraphOptimizer.Optimize(g);

        Assert.Equal(2, opt.Nodes.Count);
        Assert.Equal("Relu", opt.Nodes[0].OpType);
        Assert.Equal("X", opt.Nodes[0].Inputs[0]);       // rewired through both identities
        Assert.Equal("Identity", opt.Nodes[1].OpType);
        Assert.Equal("Y", opt.Nodes[1].Outputs[0]);

        float[] x = [-1f, 2f, -3f, 4f];
        Assert.Equal(RunSingle(g, x).Data, RunSingle(opt, x).Data);
    }

    [Fact]
    public void Optimize_RemovesDeadNodesAndInitializers()
    {
        OnnxGraph g = RunnableGraph();
        g.Initializers["unused"] = Floats("unused", [2], [9f, 9f]);        // never referenced
        g.Nodes.Add(Node("Sigmoid", ["sum"], ["deadEnd"]));                 // output unused

        OnnxGraph opt = GraphOptimizer.Optimize(g);

        Assert.Equal(2, opt.Nodes.Count);                                   // Add + Relu survive
        Assert.DoesNotContain(opt.Nodes, n => n.OpType == "Sigmoid");
        Assert.False(opt.Initializers.ContainsKey("unused"));
        Assert.True(opt.Initializers.ContainsKey("A"));

        float[] x = [1f, -2f, 3f, -4f];
        Assert.Equal(RunSingle(g, x).Data, RunSingle(opt, x).Data);
    }

    [Fact]
    public void Optimize_DoesNotMutateTheOriginalGraph()
    {
        OnnxGraph g = RunnableGraph();
        g.Nodes.Insert(0, Node("Identity", ["X"], ["alias"]));
        g.Nodes.Add(Node("Sigmoid", ["alias"], ["deadEnd"]));
        int nodes = g.Nodes.Count;
        int inits = g.Initializers.Count;

        GraphOptimizer.Optimize(g);

        Assert.Equal(nodes, g.Nodes.Count);
        Assert.Equal(inits, g.Initializers.Count);
        Assert.Equal("alias", g.Nodes[^1].Inputs[0]);   // input names untouched
    }

    [Fact]
    public void Optimize_FullyConstantGraph_StillProducesOutputs()
    {
        // Everything folds away: the output is served straight from an initializer.
        var g = new OnnxGraph { Name = "allconst", OpsetVersion = 13 };
        g.Outputs.Add(new OnnxValueInfo { Name = "Y", Shape = new[] { 2 } });
        g.Initializers["A"] = Floats("A", [2], [-1f, 3f]);
        g.Nodes.Add(Node("Relu", ["A"], ["Y"]));

        OnnxGraph opt = GraphOptimizer.Optimize(g);

        Assert.Empty(opt.Nodes);
        Assert.Equal(new[] { 0f, 3f }, Executor.Run(opt, new())["Y"].Data);
    }

    // ── Regressions from the adversarial review ───────────────────────────────

    [Fact]
    public void Read_RejectsCorruptCountsAndTruncation_WithInvalidDataException()
    {
        OnnxGraph g = RunnableGraph();
        g.Name = "";                       // fixed header: inputs count sits at byte 17
        byte[] valid;
        using (var ms = new MemoryStream())
        {
            PcnxFormat.Write(ms, g);
            valid = ms.ToArray();
        }

        byte[] negative = (byte[])valid.Clone();
        BitConverter.GetBytes(-1).CopyTo(negative, 17);
        Assert.Throws<InvalidDataException>(() => PcnxFormat.Read(negative));

        byte[] huge = (byte[])valid.Clone();
        BitConverter.GetBytes(int.MaxValue).CopyTo(huge, 17);   // > remaining bytes
        Assert.Throws<InvalidDataException>(() => PcnxFormat.Read(huge));

        Assert.Throws<InvalidDataException>(() => PcnxFormat.Read(valid[..(valid.Length / 2)]));
    }

    [Fact]
    public void RoundTrip_PreservesNullStringAndNullTensorAttributes()
    {
        // GetString falls back only on null, so null vs "" must survive a round-trip.
        var g = new OnnxGraph { Name = "nulls", OpsetVersion = 13 };
        g.Nodes.Add(Node("Fake", ["x"], ["y"],
            ("sNull", new OnnxAttribute { Type = OnnxAttributeType.String, S = null }),
            ("sEmpty", new OnnxAttribute { Type = OnnxAttributeType.String, S = "" }),
            ("tNull", new OnnxAttribute { Type = OnnxAttributeType.Tensor, T = null })));

        OnnxNode rn = RoundTrip(g).Nodes.Single();

        Assert.Null(rn.Attributes["sNull"].S);
        Assert.Equal("fallback", rn.GetString("sNull", "fallback"));
        Assert.Equal("", rn.Attributes["sEmpty"].S);
        Assert.Null(rn.Attributes["tNull"].T);
    }

    [Fact]
    public void Optimize_ElidesDropout_OnlyWhenInferenceMode()
    {
        static OnnxGraph DropoutGraph(long opset, long? isTest)
        {
            var g = new OnnxGraph { Name = "dropout", OpsetVersion = opset };
            g.Inputs.Add(new OnnxValueInfo { Name = "X", Shape = new[] { 2, 2 } });
            g.Outputs.Add(new OnnxValueInfo { Name = "Y", Shape = new[] { 2, 2 } });
            OnnxNode d = Node("Dropout", ["X"], ["mid"]);
            if (isTest is long v)
                d.Attributes["is_test"] = new OnnxAttribute { Name = "is_test", Type = OnnxAttributeType.Int, I = v };
            g.Nodes.Add(d);
            g.Nodes.Add(Node("Relu", ["mid"], ["Y"]));
            return g;
        }

        // Opset 6, training mode (is_test absent/0): NOT a passthrough — keep it
        // so Run keeps failing loudly instead of silently skipping the op.
        Assert.Contains(GraphOptimizer.Optimize(DropoutGraph(6, null)).Nodes, n => n.OpType == "Dropout");
        Assert.Contains(GraphOptimizer.Optimize(DropoutGraph(6, 0)).Nodes, n => n.OpType == "Dropout");

        // Opset 6 with is_test=1, and opset ≥ 7: inference mode → elided.
        Assert.DoesNotContain(GraphOptimizer.Optimize(DropoutGraph(6, 1)).Nodes, n => n.OpType == "Dropout");
        Assert.DoesNotContain(GraphOptimizer.Optimize(DropoutGraph(13, null)).Nodes, n => n.OpType == "Dropout");
    }

    [Fact]
    public void Run_InitializerBackedOutput_IsImmuneToCallerMutation()
    {
        var g = new OnnxGraph { Name = "allconst", OpsetVersion = 13 };
        g.Outputs.Add(new OnnxValueInfo { Name = "Y", Shape = new[] { 2 } });
        g.Initializers["A"] = Floats("A", [2], [-1f, 3f]);
        g.Nodes.Add(Node("Relu", ["A"], ["Y"]));
        OnnxGraph opt = GraphOptimizer.Optimize(g);

        Executor.Run(opt, new())["Y"].Data[1] = 999f;   // caller scribbles on the result

        Assert.Equal(new[] { 0f, 3f }, Executor.Run(opt, new())["Y"].Data);
    }

    [Fact]
    public void Save_FailureLeavesExistingFileIntact()
    {
        string path = Path.Combine(Path.GetTempPath(), $"pcnx_{Guid.NewGuid():N}.pcnx");
        try
        {
            PcnxFormat.Write(path, RunnableGraph());               // good file in place

            var bad = new OnnxGraph { Name = "bad", OpsetVersion = 13 };
            bad.Nodes.Add(Node("If", ["c"], ["y"],
                ("then_branch", new OnnxAttribute { Type = OnnxAttributeType.Graph })));

            Assert.Throws<NotSupportedException>(() => PcnxFormat.Write(path, bad));

            Assert.False(File.Exists(path + ".tmp"));               // no debris
            Assert.Equal(new[] { "Y" }, OnnxModel.Load(path).OutputNames);   // old file intact
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Run_IgnoresExtraneousInputs_BeforeAndAfterOptimize()
    {
        // An extra entry named like an internal value must not shadow it —
        // neither the node-computed value nor its constant-folded replacement.
        var g = new OnnxGraph { Name = "shadow", OpsetVersion = 13 };
        g.Inputs.Add(new OnnxValueInfo { Name = "X", Shape = new[] { 2, 2 } });
        g.Outputs.Add(new OnnxValueInfo { Name = "Y", Shape = new[] { 2, 2 } });
        g.Initializers["A"] = Floats("A", [2, 2], [1f, 2f, 3f, 4f]);
        g.Initializers["B"] = Floats("B", [2, 2], [10f, 20f, 30f, 40f]);
        g.Nodes.Add(Node("Add", ["A", "B"], ["C"]));
        g.Nodes.Add(Node("Mul", ["X", "C"], ["Y"]));
        OnnxGraph opt = GraphOptimizer.Optimize(g);

        var x = new Tensor([2, 2], [1f, 1f, 1f, 1f]);
        var shadow = new Tensor([2, 2], [-9f, -9f, -9f, -9f]);
        float[] expected = Executor.Run(g, new() { ["X"] = x })["Y"].Data;

        Assert.Equal(expected, Executor.Run(g, new() { ["X"] = x, ["C"] = shadow })["Y"].Data);
        Assert.Equal(expected, Executor.Run(opt, new() { ["X"] = x, ["C"] = shadow })["Y"].Data);
    }

    [Fact]
    public void Optimize_SkipsFolding_WhenNodeDeclaresMoreOutputsThanKernelReturns()
    {
        // Relu returns one tensor; a node declaring two outputs would be
        // half-folded, masking the loud declared-output-count error Run raises.
        var g = new OnnxGraph { Name = "twoouts", OpsetVersion = 13 };
        g.Outputs.Add(new OnnxValueInfo { Name = "a", Shape = new[] { 2 } });
        g.Initializers["A"] = Floats("A", [2], [-1f, 3f]);
        g.Nodes.Add(Node("Relu", ["A"], ["a", "b"]));

        OnnxGraph opt = GraphOptimizer.Optimize(g);

        Assert.Single(opt.Nodes);                       // kept, not folded
        Assert.False(opt.Initializers.ContainsKey("a"));
    }
}

/// <summary>
/// PCNX + optimizer against the real downloaded models: the optimized, saved
/// and reloaded model must produce bit-identical outputs to the freshly parsed
/// .onnx (folding uses the same kernels, so not even float noise is allowed).
/// Each test skips silently when its model is not on disk.
/// </summary>
public class OnnxPcnxIntegrationTests
{
    private static readonly string ModelsDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".predictivecoding", "vision-models");

    [Fact]
    [Trait("Category", "Integration")]
    public void Yolox_OptimizedPcnx_RunsBitIdentically()
        => AssertPcnxParity("yolox_2022nov.onnx", [1, 3, 640, 640]);

    [Fact]
    [Trait("Category", "Integration")]
    public void RetinaFace_OptimizedPcnx_RunsBitIdentically()
        => AssertPcnxParity("det_10g.onnx", [1, 3, 640, 640]);

    [Fact]
    [Trait("Category", "Integration")]
    public void ArcFace_OptimizedPcnx_RunsBitIdentically()
        => AssertPcnxParity("arcface_w600k_r50.onnx", [1, 3, 112, 112]);

    private static void AssertPcnxParity(string fileName, int[] shape)
    {
        string path = Path.Combine(ModelsDir, fileName);
        if (!File.Exists(path)) return;   // model not downloaded on this machine

        var rng = new Random(42);
        var data = new float[Tensor.CheckedProduct(shape)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        OnnxModel original = OnnxModel.Load(path);
        Dictionary<string, Tensor> expected =
            original.Run(new() { [original.InputNames[0]] = new Tensor(shape, data) });

        string pcnxPath = Path.Combine(Path.GetTempPath(), $"{fileName}_{Guid.NewGuid():N}.pcnx");
        try
        {
            original.Save(pcnxPath);   // optimized save
            OnnxModel reloaded = OnnxModel.Load(pcnxPath);

            Assert.True(reloaded.Graph.Nodes.Count <= original.Graph.Nodes.Count,
                $"optimizer grew the graph: {original.Graph.Nodes.Count} → {reloaded.Graph.Nodes.Count} nodes");
            Assert.Equal(original.OutputNames, reloaded.OutputNames);

            Dictionary<string, Tensor> actual =
                reloaded.Run(new() { [reloaded.InputNames[0]] = new Tensor(shape, data) });

            foreach ((string name, Tensor exp) in expected)
            {
                Assert.True(actual.ContainsKey(name), $"reloaded model missing output '{name}'");
                Assert.Equal(exp.Shape, actual[name].Shape);
                Assert.Equal(exp.Data, actual[name].Data);       // bit-identical, no tolerance
            }
        }
        finally
        {
            File.Delete(pcnxPath);
        }
    }
}
