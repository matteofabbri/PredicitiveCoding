namespace PredictiveCoding.Benchmark;

/// <summary>The kind of evaluation a face dataset supports.</summary>
public enum FaceTaskKind
{
    /// <summary>1:1 — are these two faces the same person? (LFW, RFW, IJB-C)</summary>
    Verification,
    /// <summary>1:N — which gallery identity is this probe? (CASIA, VGGFace2, CelebA, DigiFace, Glint360K)</summary>
    Identification,
    /// <summary>Demographic / attribute classification (FairFace).</summary>
    Attribute,
}
