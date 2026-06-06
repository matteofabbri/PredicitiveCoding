namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// IEnvironment  —  embodiment interface for the sensorimotor loop.
//
// Abstracts the physical (or simulated) world that the agent explores with a
// moveable sensor.  Each step the agent observes a (feature, location) pair
// from the current sensor position and can issue a motor command to relocate
// the sensor for the next observation.
//
// Implementations can represent tactile exploration, visual scanning, or any
// process where a sensor moves relative to an object and reports local features.
//
// Reference: Leadholm et al. (2025) P9 §3 — "sensorimotor loop".
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Sensorimotor environment abstraction for the agent loop.</summary>
public interface IEnvironment
{
    /// <summary>Total number of objects registered in this environment.</summary>
    int ObjectCount { get; }

    /// <summary>The id of the object the sensor is currently exploring.
    /// Returns -1 when no object has been presented yet.</summary>
    int CurrentObjectId { get; }

    /// <summary>The number of distinct motor commands available.
    /// <see cref="MoveTo"/> accepts values in [0, AvailableActions).</summary>
    int AvailableActions { get; }

    /// <summary>Positions the sensor at the beginning of <paramref name="objectId"/>
    /// and readies the environment for a new exploration episode.</summary>
    void PresentObject(int objectId);

    /// <summary>Returns the (Feature, Location) SDR pair at the current sensor position.</summary>
    (Sdr Feature, Sdr Location) Observe();

    /// <summary>Executes a motor command and advances the sensor to the next position.
    /// <paramref name="motorCommand"/> must be in [0, <see cref="AvailableActions"/>).</summary>
    void MoveTo(int motorCommand);

    /// <summary>Resets the sensor to its initial position on the current object
    /// without changing which object is being explored.</summary>
    void Reset();
}
