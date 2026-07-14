using PredictiveCoding.Imaging;
using PredictiveCoding.Vision;
using PredictiveCoding.Vision.Capture;
using PredictiveCoding.Vision.Detectors;
using PredictiveCoding.Vision.Tracking;

namespace PredictiveCoding.Main.Demos;

// ════════════════════════════════════════════════════════════════════════════
// PeopleCountingDemo
//
//   People counting on a camera feed or video file — docs/10-people-counting-roadmap.md
//   Phases 0–3 (capture → detect → track → count).
//
//     • Source   : webcam index ("0"), a video-file path, or an rtsp:// URL.
//     • Detector : YOLOX full-body (COCO "person", Apache-2.0, auto-downloaded)
//                  or RetinaFace face/head count (already bundled, no download).
//     • Tracking : a SORT-style tracker turns per-frame boxes into stable IDs, so
//                  the same person is not counted twice.
//     • Output   : live "now" (people visible) + "unique" (distinct people seen)
//                  in the console; optionally saves frames with per-id coloured boxes.
//
//   Line-crossing / zone counting (Phase 4) is the next step.
// ════════════════════════════════════════════════════════════════════════════
public static class PeopleCountingDemo
{
    private static readonly string ModelsDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".predictivecoding", "vision-models");

    private static readonly string OutputDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".predictivecoding", "vision-output");

    public static void Run()
    {
        Console.WriteLine("\n  ── People counting (detect · track · count) ────────────────");
        Console.WriteLine("  Counts people in a live camera feed or a video file.\n");

        Console.Write("  Source — webcam index, video path, or rtsp:// URL [0]: ");
        string source = Console.ReadLine()?.Trim() is { Length: > 0 } s ? s : "0";

        Console.WriteLine("\n  Detector:");
        Console.WriteLine("    1  YOLOX full-body  (COCO 'person', Apache-2.0, downloads ~36 MB on first use)");
        Console.WriteLine("    2  RetinaFace faces (already bundled, counts faces pointing at the camera)");
        Console.Write("  Choice [1]: ");
        string det = Console.ReadLine()?.Trim() is { Length: > 0 } d ? d : "1";

        Console.Write("\n  Process every Nth frame (1 = every frame) [2]: ");
        int stride = int.TryParse(Console.ReadLine(), out int st) && st > 0 ? st : 2;

        Console.Write("  Max frames to process (0 = until stopped / EOF) [0]: ");
        int maxFrames = int.TryParse(Console.ReadLine(), out int mf) && mf > 0 ? mf : 0;

        Console.Write("  Save frames with tracked boxes to disk? [y/N]: ");
        bool save = Console.ReadLine()?.Trim().StartsWith("y", StringComparison.OrdinalIgnoreCase) == true;

        using IPersonDetector detector = det == "2"
            ? new FaceCountDetector(ModelsDir)
            : new YoloPersonDetector(ModelsDir);

        var tracker = new SortTracker();

        Console.WriteLine($"\n  Detector : {detector.Name}");
        Console.WriteLine($"  Source   : {source}");
        if (save)
        {
            Directory.CreateDirectory(OutputDir);
            Console.WriteLine($"  Saving   : {OutputDir}");
        }
        Console.WriteLine("\n  Warming up the detector… (first frame also triggers any model download)");
        Console.WriteLine("  Press any key to stop.\n");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        int frame = 0, peakNow = 0;

        try
        {
            foreach (RgbImage img in FrameStream.Read(source, stride))
            {
                using (img)
                {
                    IReadOnlyList<Detection> dets = detector.Detect(img);
                    IReadOnlyList<Track> active = tracker.Update(dets);
                    int now = tracker.CurrentCount;
                    int unique = tracker.UniqueCount;
                    int raw = dets.Count;
                    peakNow = Math.Max(peakNow, now);
                    frame++;

                    double fps = frame / Math.Max(1e-3, sw.Elapsed.TotalSeconds);
                    Console.Write(
                        $"\r  frame {frame,5} │ now {now,3} │ unique {unique,4} │ raw {raw,3} │ peak {peakNow,3} │ {fps,5:F1} fps   ");

                    if (save)
                    {
                        Annotate.DrawTracks(img, active);
                        img.SaveAsJpeg(Path.Combine(OutputDir, $"frame_{frame:D5}_now{now}.jpg"));
                    }
                }

                if (maxFrames > 0 && frame >= maxFrames) break;
                if (KeyPressed()) break;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n\n  [stopped] {ex.Message}");
            Console.WriteLine("  Tip: for a webcam try index 0; for a file pass a full path; for an IP cam an rtsp:// URL.");
        }

        Console.WriteLine("\n");
        Console.WriteLine($"  Frames processed      : {frame}");
        Console.WriteLine($"  Unique people (total) : {tracker.UniqueCount}");
        Console.WriteLine($"  Peak concurrent       : {peakNow}");
        Console.WriteLine("\n  Done.");
    }

    // Non-blocking "any key" check; harmless no-op when stdin is redirected.
    private static bool KeyPressed()
    {
        try
        {
            if (!Console.KeyAvailable) return false;
            Console.ReadKey(intercept: true);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
