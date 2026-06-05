using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Interaction;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Midi;

/// <summary>A note event: pitch, timing (in ticks) and velocity.</summary>
public sealed record MidiNote(int NoteNumber, long Start, long Length, int Velocity);

/// <summary>A parsed MIDI file: its notes and tick resolution.</summary>
public sealed class MidiData
{
    /// <summary>The note events, ordered as read.</summary>
    public IReadOnlyList<MidiNote> Notes { get; }

    /// <summary>Ticks per quarter note (timing resolution).</summary>
    public short TicksPerQuarterNote { get; }

    public MidiData(IReadOnlyList<MidiNote> notes, short ticksPerQuarterNote)
    {
        ArgumentNullException.ThrowIfNull(notes);
        Notes = notes;
        TicksPerQuarterNote = ticksPerQuarterNote;
    }

    /// <summary>
    /// Renders the notes as a piano-roll tensor <c>[timeSteps, 128]</c>, where each
    /// time step spans <paramref name="resolutionTicks"/> ticks and a cell holds the
    /// note velocity (0..1) while that pitch is sounding.
    /// </summary>
    public DenseTensor<float> ToPianoRoll(int resolutionTicks = 120)
    {
        if (resolutionTicks <= 0)
            throw new ArgumentOutOfRangeException(nameof(resolutionTicks));

        long maxEnd = 0;
        foreach (MidiNote note in Notes)
            maxEnd = Math.Max(maxEnd, note.Start + note.Length);

        int steps = maxEnd > 0 ? (int)((maxEnd + resolutionTicks - 1) / resolutionTicks) : 1;
        var data = new float[steps * 128];

        foreach (MidiNote note in Notes)
        {
            if (note.NoteNumber is < 0 or > 127)
                continue;
            int start = (int)(note.Start / resolutionTicks);
            int end = (int)((note.Start + Math.Max(0, note.Length - 1)) / resolutionTicks);
            for (int step = start; step <= end && step < steps; step++)
                data[step * 128 + note.NoteNumber] = note.Velocity / 127f;
        }

        return new DenseTensor<float>(data, steps, 128);
    }
}

/// <summary>Reads MIDI files with DryWetMidi.</summary>
public static class MidiReader
{
    /// <summary>Reads a MIDI file from disk.</summary>
    public static MidiData Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("MIDI file not found.", path);
        return Convert(MidiFile.Read(path));
    }

    /// <summary>Reads a MIDI file from a stream.</summary>
    public static MidiData Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        return Convert(MidiFile.Read(stream));
    }

    private static MidiData Convert(MidiFile midiFile)
    {
        var notes = new List<MidiNote>();
        foreach (Note note in midiFile.GetNotes())
            notes.Add(new MidiNote((byte)note.NoteNumber, note.Time, note.Length, (byte)note.Velocity));

        short ticksPerQuarterNote = midiFile.TimeDivision is TicksPerQuarterNoteTimeDivision division
            ? division.TicksPerQuarterNote
            : (short)480;
        return new MidiData(notes, ticksPerQuarterNote);
    }
}
