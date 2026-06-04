using System.Text;

namespace PredictiveCoding.LLM.Generation;

// ─────────────────────────────────────────────────────────────────────────────
// ChatTemplate  —  formats chat turns into a single prompt string.
//
// A minimal role-tagged template (system / user / assistant).  The result is fed
// to the tokenizer like any other text; the trailing assistant tag cues the model
// to continue as the assistant.
// ─────────────────────────────────────────────────────────────────────────────

public static class ChatTemplate
{
    public sealed record Turn(string Role, string Content);

    /// <summary>Formats a conversation, leaving the assistant turn open for generation.</summary>
    public static string Format(IEnumerable<Turn> turns, string? system = null)
    {
        var sb = new StringBuilder();
        if (!string.IsNullOrEmpty(system))
            sb.Append("<|system|>\n").Append(system).Append('\n');
        foreach (var t in turns)
            sb.Append("<|").Append(t.Role).Append("|>\n").Append(t.Content).Append('\n');
        sb.Append("<|assistant|>\n");
        return sb.ToString();
    }

    /// <summary>Convenience overload for a single user message.</summary>
    public static string Format(string userMessage, string? system = null)
        => Format([new Turn("user", userMessage)], system);
}
