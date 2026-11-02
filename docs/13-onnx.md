# 13. Onnx — `PredictiveCoding.Onnx`

An **in-house ONNX runtime**: it loads `.onnx` models, executes them, optimises
the graph and re-saves them in a compact native format — with **no ONNX
Runtime, no protobuf package, no downloads**. The project is deliberately
dependency-free; the whole point is that the models this repo ships can run
anywhere the .NET runtime runs.

It exists because the vision stack needs pretrained deep models (YOLOX person
detection, ArcFace / RetinaFace-class face models — see
[doc 10](10-people-counting-roadmap.md) and
[doc 15](15-face-recognition.md)) and the solution refuses to make a heavy
native runtime a hard dependency. The heavy `Microsoft.ML.OnnxRuntime` package
remains available for speed; this project is the always-there fallback and the
optimiser.

## The pipeline

```
model.onnx ──ProtoReader──► OnnxParser ──► OnnxGraph ──► Executor ──► outputs
                (hand-written protobuf reader)              │
                                                            ▼
                                     Ops.* kernels (Conv, MatMul, …)
```

| Folder | Types | Role |
|--------|-------|------|
| `Proto/` | `ProtoReader`, `OnnxParser`, `OnnxGraph`, `OnnxNode`, … | A hand-written **protobuf wire-format reader** (varint/tag decoding) parses the model file directly — no protobuf library. External-data side files (`model.onnx_data`) resolve relative to the model's directory. |
| root | `Ops.*.cs` (`Conv`, `Elementwise`, `Gather`, `Linear`, `Logical`, `Pool`, `Reduce`, `Resize`, `Shape`) | The kernel set, organised by family. Intentionally small: enough for the models this repo ships; unsupported ops **fail loudly at Run** instead of silently degrading. |
| root | `Executor`, `Tensor` | Walks the graph in topological order, feeding initializers and calling the kernels. `Tensor` is the runtime's own ND value type. |
| `Optimize/` | `GraphOptimizer` | **Constant folding** (evaluated with the runtime's own kernels), **Identity elision**, **dead-code removal** — all provably result-preserving (`Run` output stays bit-identical). |
| `Format/` | `PcnxFormat` | **PCNX**, the in-house save format: streams from disk (files may exceed the 2 GB protobuf limit), sniffs its own magic on load. |

## `OnnxModel` — the public API

```csharp
var model  = OnnxModel.Load("yolox.onnx");      // .onnx protobuf or PCNX (sniffed)
var output = model.Run(new() { ["images"] = inputTensor })["output"];
model.Save("yolox.pcnx");                        // optimised in-house format
var again  = OnnxModel.Load("yolox.pcnx");       // same API, faster load
```

- `Load` accepts both formats from content sniffing (extension irrelevant);
  a `.onnx` file over 2 GB is rejected with a clear error (protobuf's own
  limit), while PCNX files that large are fine.
- `Save(path, optimize: true)` runs the `GraphOptimizer` first;
  `Optimized()` returns an optimised in-memory copy.
- `InputNames` / `OutputNames` name what a caller must feed and gets back.
- `Run` returns every declared graph output by name.

## PCNX & the OnnxPack tool

`.pcnx` is what you check into the repo or ship with an app: no protobuf parse
at load, weights streamed from disk, and the graph already folded. Convert
models with the [`OnnxPack` tool](16-tools-and-cli.md):

```bash
dotnet run --project tools/OnnxPack -- model.onnx     # → model.pcnx (optimised)
```

## Tests

The `Onnx*Tests` files in the unit-test suite cover the kernels
(`OnnxOpsTests`, `OnnxTransformerOpsTests`, `OnnxResizeTests`), the parser and
external data (`OnnxParserTests`, `OnnxExternalDataTests`), the PCNX round-trip
(`OnnxPcnxFormatTests`, `OnnxPcnxIntegrationTests`), the optimiser's
bit-identical guarantee (`OnnxReviewFixTests`) and the detector wiring
(`OnnxDetectorWiringTests`).
