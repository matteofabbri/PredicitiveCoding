using PredictiveCoding.LLM.Generation;

namespace PredictiveCoding.LLM;

// ─────────────────────────────────────────────────────────────────────────────
// ChatSession  —  multi-turn chat on a PredictiveLlm (Pro roadmap Phase 2).
//
// History is kept as role-tagged turns and rendered through ChatTemplate on
// every exchange; replies stop at the next role tag (or any caller-supplied
// stop sequence). The self-feeding rule extends to chat: with learnFromUser
// the model absorbs ONLY the user's words — never the formatted history and
// never its own assistant turns (learning from your own output collapses).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ChatSession
{
    private readonly PredictiveLlm _llm;
    private readonly string? _system;
    private readonly List<ChatTemplate.Turn> _turns = [];

    /// <summary>The conversation so far, user and assistant turns alternating.</summary>
    public IReadOnlyList<ChatTemplate.Turn> History => _turns;

    internal ChatSession(PredictiveLlm llm, string? system)
    {
        _llm    = llm;
        _system = system;
    }

    /// <summary>
    /// One exchange: the user speaks, the assistant answers, both land in the
    /// history. <paramref name="learnFromUser"/> absorbs the user's message
    /// (one online <see cref="PredictiveLlm.Learn"/> step) before answering.
    /// Throws when the rendered history outgrows the model's window — trim or
    /// start a new session (history compression is a later phase).
    /// </summary>
    public string Say(string userMessage, int maxNewTokens = 64, GenerationOptions? options = null,
                      bool learnFromUser = false)
    {
        if (learnFromUser && userMessage.Length >= 2)
            _llm.Learn(userMessage);

        _turns.Add(new ChatTemplate.Turn("user", userMessage));
        string prompt = ChatTemplate.Format(_turns, _system);

        var opts = (options ?? new GenerationOptions()) with
        {
            LearnFromPrompt = false,                       // never absorb the formatted history
            StopSequences   = [.. options?.StopSequences ?? [], "<|"],   // stop at the next role tag
        };

        string reply = _llm.GenerateDetailed(prompt, maxNewTokens, opts).Text.TrimEnd('\n', '\r');
        _turns.Add(new ChatTemplate.Turn("assistant", reply));
        return reply;
    }
}
