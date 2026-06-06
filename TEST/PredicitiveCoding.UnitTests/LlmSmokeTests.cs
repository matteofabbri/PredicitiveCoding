using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

public class LlmSmokeTests
{
    private static GptModel TinyModel() => new(
        new TransformerConfig
        {
            VocabSize = 260, MaxSeqLen = 32,
            EmbedDim = 32, NumHeads = 2, NumLayers = 2, FeedForwardDim = 64,
        },
        new Random(42));

    [Fact]
    public void ByteTokenizer_RoundTrips()
    {
        var tok = new ByteTokenizer();
        var ids = tok.Encode("Hello, world!", addBos: true, addEos: true);
        Assert.Equal(tok.Bos, ids[0]);
        Assert.Equal(tok.Eos, ids[^1]);
        Assert.Equal("Hello, world!", tok.Decode(ids));   // specials skipped
    }

    [Fact]
    public void Step_ReturnsVocabSizedLogits()
    {
        var model = TinyModel();
        var cache = model.CreateCache();
        var logits = model.Step(token: 65, position: 0, cache);
        Assert.Equal(model.VocabSize, logits.Length);
        Assert.All(logits.Buffer, f => Assert.True(float.IsFinite(f)));
    }

    [Fact]
    public void Generator_ProducesTokensInRange()
    {
        var model = TinyModel();
        var tok   = new ByteTokenizer();
        var gen   = new TextGenerator(model, tok);

        var opts   = new SamplingOptions { MaxNewTokens = 16, Temperature = 1f, TopK = 40, Seed = 7 };
        var output = gen.GenerateTokens(tok.Encode("hi", addBos: true), opts);

        Assert.True(output.Count <= 16);
        Assert.All(output, t => Assert.InRange(t, 0, model.VocabSize - 1));
    }

    [Fact]
    public void Generator_GreedyIsDeterministic()
    {
        var tok  = new ByteTokenizer();
        var opts = new SamplingOptions { MaxNewTokens = 8, Greedy = true };

        var a = new TextGenerator(TinyModel(), tok).GenerateTokens(tok.Encode("x", addBos: true), opts);
        var b = new TextGenerator(TinyModel(), tok).GenerateTokens(tok.Encode("x", addBos: true), opts);
        Assert.Equal(a, b);
    }

    [Fact]
    public void CrossEntropy_PerplexityIsFiniteAndAtLeastOne()
    {
        var model = TinyModel();
        var tok   = new ByteTokenizer();
        var (loss, ppl) = CrossEntropy.Evaluate(model, tok.Encode("predictive coding"));
        Assert.True(float.IsFinite(loss) && loss >= 0f);
        Assert.True(ppl >= 1f);
    }

    [Fact]
    public void TextDataset_ProducesShiftedWindows()
    {
        var tok = new ByteTokenizer();
        var ds  = new TextDataset("abcdefghij", tok, seqLen: 4);
        var (input, target) = ds.Windows(stride: 2).First();

        Assert.Equal(4, input.Length);
        Assert.Equal(4, target.Length);
        for (int i = 0; i < 3; i++) Assert.Equal(input[i + 1], target[i]);  // target = input shifted by 1
    }
}
