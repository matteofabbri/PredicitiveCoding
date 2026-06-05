using Melanchall.DryWetMidi.Common;
using Melanchall.DryWetMidi.Core;
using PredictiveCoding.Core;
using PredictiveCoding.Data.Midi;
// Disambiguate from Melanchall.DryWetMidi.Core.MidiReader.
using DataMidiReader = PredictiveCoding.Data.Midi.MidiReader;

namespace PredicitiveCoding.UnitTests;

public sealed class MidiReaderTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_midi_{Guid.NewGuid():N}");

    public MidiReaderTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    private string WriteMidi()
    {
        var track = new TrackChunk(
            new NoteOnEvent((SevenBitNumber)(byte)60, (SevenBitNumber)(byte)100) { DeltaTime = 0 },
            new NoteOffEvent((SevenBitNumber)(byte)60, (SevenBitNumber)(byte)0) { DeltaTime = 480 });
        var midiFile = new MidiFile(track);
        string path = Path.Combine(_tmp, "song.mid");
        midiFile.Write(path);
        return path;
    }

    [Fact]
    public void Read_ReadsNote()
    {
        MidiData data = DataMidiReader.Read(WriteMidi());

        MidiNote note = Assert.Single(data.Notes);
        Assert.Equal(60, note.NoteNumber);
        Assert.Equal(100, note.Velocity);
        Assert.Equal(0, note.Start);
        Assert.Equal(480, note.Length);
    }

    [Fact]
    public void ToPianoRoll_PlacesNoteOnGrid()
    {
        MidiData data = DataMidiReader.Read(WriteMidi());

        DenseTensor<float> roll = data.ToPianoRoll(resolutionTicks: 120);

        Assert.Equal(2, roll.Rank);
        Assert.Equal(128, roll.Dimensions[1]);
        Assert.True(roll.Dimensions[0] >= 4);    // 480 ticks / 120 = 4 steps
        Assert.True(roll[0, 60] > 0f);           // note sounding at step 0, pitch 60
        Assert.Equal(0f, roll[0, 61]);           // a silent pitch
    }

    [Fact]
    public void Read_Missing_Throws()
        => Assert.Throws<FileNotFoundException>(() => DataMidiReader.Read(Path.Combine(_tmp, "nope.mid")));
}
