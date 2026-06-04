using PredictiveCoding.LLM.Generation;

namespace PredicitiveCoding.UnitTests;

public class ChatTemplateTests
{
    [Fact]
    public void Format_SingleUserMessage_OpensAssistantTurn()
    {
        string s = ChatTemplate.Format("hello");
        Assert.Contains("<|user|>", s);
        Assert.Contains("hello", s);
        Assert.EndsWith("<|assistant|>\n", s);
    }

    [Fact]
    public void Format_WithSystemAndTurns_IncludesRoles()
    {
        string s = ChatTemplate.Format(
            [new ChatTemplate.Turn("user", "hi"), new ChatTemplate.Turn("assistant", "hey"), new ChatTemplate.Turn("user", "bye")],
            system: "be terse");

        Assert.Contains("<|system|>", s);
        Assert.Contains("be terse", s);
        Assert.Contains("hey", s);
        Assert.EndsWith("<|assistant|>\n", s);
    }
}
