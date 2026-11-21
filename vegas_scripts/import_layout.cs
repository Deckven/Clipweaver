/**
 * Vegas Pro 22: universal layout loader.
 *
 * One script for every project. Reads a layout.txt produced by
 * `python -m clipmaker.layout` and places the events it describes.
 * Nothing is decided here — positions, lengths and take offsets were
 * computed and validated in Python; this file only executes the plan.
 *
 * Sync model: every time in layout.txt is relative to the narration start.
 * The script finds the narration audio event on the timeline (by the file
 * name from the layout header) and adds its position — so wherever the mp3
 * sits, the video follows it. No hardcoded offsets.
 *
 * Raw takes instead of the mix (v5): when the timeline holds the raw
 * narration takes (Recording N.mp3, Changes N.mp3, ...) rather than the
 * rendered mp3, the layout's take= header lines (from the project's
 * vegas.json) say where each piece of each take sits in the narration. Every
 * take event on the timeline then implies a narration zero — event start -
 * take offset - lag — and the median wins. This replaces the per-project
 * copies of this script with hand-typed SEG_* / REC_* lag tables. Only when
 * neither the mp3 nor a listed take is found does the script offer timeline
 * 00:00:00.
 *
 * Overlays (v5): if overlays\layout.txt exists next to the picked layout.txt
 * (number plaques, text cards), it is placed as well, on its own track at the
 * TOP of the track list so its transparent images composite over the B-roll,
 * with the same narration anchor.
 *
 * Times travel as integer milliseconds and become Timecode.FromMilliseconds:
 * no fps guessing on this side. Python already quantized them to whole
 * frames at the declared fps; the script only warns when the project's
 * frame rate differs from the layout's.
 *
 * Usage:
 *   1. Open the project; the narration mp3 or its raw takes must be on the
 *      timeline.
 *   2. Tools -> Scripting -> Run Script -> import_layout.cs
 *   3. Pick the project's layout.txt in the file dialog.
 *   4. Confirm the summary. A track with the layout's name appears (and the
 *      overlay track, if any); existing tracks with those names are
 *      replaced, not stacked.
 */

using System;
using System.IO;
using System.Collections.Generic;
using System.Globalization;
using System.Windows.Forms;
using ScriptPortal.Vegas;

public class EntryPoint
{
    // Folder the file dialog opens in: Clipmaker's projects folder, from the
    // CLIPMAKER_PROJECTS_DIR environment variable (a Windows user variable,
    // read when Vegas starts). Unset: the dialog opens where it last was.
    static readonly string LAYOUTS_DIR =
        Environment.GetEnvironmentVariable("CLIPMAKER_PROJECTS_DIR") ?? "";

    // Optional pin: set the path to a layout.txt here to skip the file
    // dialog. Empty in the generic loader — do not commit a pinned copy.
    const string LAYOUT_FILE = @"";

    // Second layout picked up next to the first one, placed on a top track.
    const string OVERLAY_SUBPATH = @"overlays\layout.txt";

    // Bumped on every change; shown in dialog titles and written to the log so
    // a stale cached compile is immediately visible.
    const string VERSION = "v6";

    // Every run appends here, even when no dialog is shown — if this file does
    // not grow, the script did not execute at all (Vegas cache / compile fail).
    // CLIPMAKER_VEGAS_LOG overrides the default %TEMP%\clipmaker_import_layout.log.
    static readonly string LOG_FILE =
        Environment.GetEnvironmentVariable("CLIPMAKER_VEGAS_LOG")
        ?? System.IO.Path.Combine(System.IO.Path.GetTempPath(), "clipmaker_import_layout.log");

    // A take piece whose implied narration zero is further than this from the
    // median was moved after the mix was rendered.
    public const double AGREE_MS = 50.0;

    static readonly CultureInfo CI = CultureInfo.InvariantCulture;

    static void Log(string msg)
    {
        try
        {
            File.AppendAllText(LOG_FILE,
                DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " [" + VERSION + "] "
                + msg + "\r\n");
        }
        catch { }
    }

    public void FromVegas(Vegas vegas)
    {
        Log("=== script started, project: " + vegas.Project.FilePath);
        try { Run(vegas); }
        catch (Exception ex)
        {
            Log("EXCEPTION: " + ex.GetType().Name + ": " + ex.Message);
            // Without this Vegas shows a bare "Error 0x80131600".
            MessageBox.Show(ex.GetType().Name + ": " + ex.Message +
                            "\n\n" + ex.StackTrace, "Layout import failed " + VERSION);
        }
        Log("=== script finished");
    }

    void Run(Vegas vegas)
    {
        string layoutPath = PickLayoutFile();
        if (layoutPath == null) { Log("no layout picked, aborting"); return; }
        Log("layout: " + layoutPath);

        Layout layout = ParseLayout(layoutPath);
        if (layout.Events.Count == 0)
        {
            MessageBox.Show("No events in " + layoutPath);
            return;
        }

        Timecode audioStart = FindNarrationStart(vegas, layout.Audio);
        Log("narration \"" + layout.Audio + "\" found at: " + (audioStart == null ? "NOT FOUND" : audioStart.ToString()));
        string anchorNote = "";
        if (audioStart == null && layout.Takes.Count > 0)
        {
            string detail;
            audioStart = FindStartFromTakes(vegas, layout.Takes, out detail);
            Log("anchor from raw takes (" + layout.Takes.Count + " take segments): " +
                (audioStart == null ? "NONE" : audioStart.ToString()));
            Log(detail);
            if (audioStart != null)
                anchorNote = "\n(anchored on the raw takes)\n" + detail;
        }
        if (audioStart == null)
        {
            Log("audio on timeline (multi-line):");
            Log(ListAudioEvents(vegas));
            // The narration may be on the timeline as raw recording pieces
            // the layout has no take= lines for. If the merged file was
            // rendered from timeline zero, syncing to 00:00:00 is equivalent —
            // offer that instead of giving up.
            DialogResult fallback = MessageBox.Show(
                "Narration event not found on the timeline:\n  " + layout.Audio +
                (layout.Takes.Count > 0 ? "\n(and none of the layout's raw takes either)" : "") +
                "\n\nAudio events currently on the timeline:\n" +
                ListAudioEvents(vegas) +
                "\n\nPlace events relative to timeline 00:00:00 instead?\n" +
                "(Correct when the narration mix starts at project zero. " +
                "To sync to the mp3 instead, press Cancel, put " + layout.Audio +
                " on a muted track and rerun.)",
                "Narration not found — use timeline zero? " + VERSION,
                MessageBoxButtons.OKCancel);
            Log("timeline-zero fallback: " + fallback);
            if (fallback != DialogResult.OK) return;
            audioStart = Timecode.FromMilliseconds(0);
        }

        double projectFps = vegas.Project.Video.FrameRate;
        string fpsNote = Math.Abs(projectFps - layout.Fps) < 0.01 ? ""
            : "\n\nWARNING: layout was quantized at " + layout.Fps +
              " fps but the project runs at " + projectFps + " fps.";

        string overlayPath = Path.Combine(Path.GetDirectoryName(layoutPath), OVERLAY_SUBPATH);
        Layout overlay = File.Exists(overlayPath) ? ParseLayout(overlayPath) : null;
        string overlayNote = "";
        if (overlay != null && overlay.Track == layout.Track)
        {
            overlayNote = "\n\nOverlays skipped: their track name equals the B-roll track (" +
                          layout.Track + ")";
            overlay = null;
        }
        else if (overlay != null)
        {
            overlayNote = "\n\nOverlays: " + overlay.Events.Count + " on track \"" +
                          overlay.Track + "\" at the top";
        }
        Log("overlay layout: " + (overlay == null ? "none (" + overlayPath + ")"
                                                  : overlay.Events.Count + " events"));

        DialogResult ok = MessageBox.Show(
            "Layout: " + Path.GetFileName(layoutPath) +
            "\nProject: " + layout.Project +
            "\nEvents: " + layout.Events.Count +
            "\nNarration found at: " + audioStart.ToString() + anchorNote +
            "\nTrack: " + layout.Track + " (existing one will be replaced)" +
            overlayNote + fpsNote + "\n\nPlace events?",
            "Confirm layout import " + VERSION, MessageBoxButtons.OKCancel);
        Log("confirm dialog: " + ok);
        if (ok != DialogResult.OK) return;

        string report = PlaceLayout(vegas, layout, audioStart, false);
        if (overlay != null)
            report += "\n\n" + PlaceLayout(vegas, overlay, audioStart, true);
        report += "\n\nSynced to narration at " + audioStart.ToString();
        Log(report.Replace("\n", " | "));
        MessageBox.Show(report, "Layout import " + VERSION);
    }

    // Places one layout on its own track. atTop: insert the track above all
    // others so its (transparent) events composite over the B-roll.
    string PlaceLayout(Vegas vegas, Layout layout, Timecode audioStart, bool atTop)
    {
        RemoveTracksNamed(vegas, layout.Track);
        VideoTrack track;
        if (atTop)
        {
            track = new VideoTrack(vegas.Project, 0, layout.Track);
            vegas.Project.Tracks.Add(track);
        }
        else
        {
            track = vegas.Project.AddVideoTrack();
            track.Name = layout.Track;
        }

        int placed = 0;
        var problems = new List<string>();

        foreach (Ev e in layout.Events)
        {
            if (!File.Exists(e.Path))
            {
                problems.Add("missing file: " + Path.GetFileName(e.Path));
                continue;
            }
            try
            {
                Media media = vegas.Project.MediaPool.AddMedia(e.Path);
                MediaStream stream = (media == null) ? null : media.GetVideoStreamByIndex(0);
                if (stream == null)
                {
                    problems.Add(Path.GetFileName(e.Path) + " — no video stream");
                    continue;
                }
                VideoEvent ev = track.AddVideoEvent(
                    audioStart + Timecode.FromMilliseconds(e.StartMs),
                    Timecode.FromMilliseconds(e.LengthMs));
                Take take = ev.AddTake(stream);
                take.Offset = Timecode.FromMilliseconds(e.TakeOffsetMs);

                double fadeMs = Math.Min(layout.FadeMs, e.LengthMs / 2.0);
                if (fadeMs > 0)
                {
                    ev.FadeIn.Length = Timecode.FromMilliseconds(fadeMs);
                    ev.FadeOut.Length = Timecode.FromMilliseconds(fadeMs);
                }
                placed++;
            }
            catch (Exception ex)
            {
                problems.Add(Path.GetFileName(e.Path) + " — " + ex.Message);
            }
        }

        string report = "Track \"" + layout.Track + "\": placed " + placed +
                        " of " + layout.Events.Count;
        if (problems.Count > 0)
        {
            report += "\nProblems:\n";
            foreach (string s in problems) report += "  " + s + "\n";
        }
        return report;
    }

    // --- narration lookup -------------------------------------------------

    string ListAudioEvents(Vegas vegas)
    {
        var lines = new List<string>();
        foreach (Track t in vegas.Project.Tracks)
        {
            if (!t.IsAudio()) continue;
            foreach (TrackEvent ev in t.Events)
            {
                Take take = ev.ActiveTake;
                string name = (take != null && take.Media != null)
                    ? Path.GetFileName(take.Media.FilePath) : "(no media)";
                lines.Add("  " + ev.Start.ToString() + "  " + name);
                if (lines.Count >= 20) { lines.Add("  ..."); break; }
            }
            if (lines.Count >= 20) break;
        }
        return lines.Count > 0 ? string.Join("\n", lines.ToArray())
                               : "  (none — no audio events at all)";
    }

    Timecode FindNarrationStart(Vegas vegas, string audioFileName)
    {
        string wanted = audioFileName.ToLowerInvariant();
        foreach (Track t in vegas.Project.Tracks)
        {
            if (!t.IsAudio()) continue;
            foreach (TrackEvent ev in t.Events)
            {
                Take take = ev.ActiveTake;
                if (take == null || take.Media == null) continue;
                string file = Path.GetFileName(take.Media.FilePath).ToLowerInvariant();
                if (file == wanted) return ev.Start;
            }
        }
        return null;
    }

    // Collects every audio event and hands it to AnchorFromTakes.
    Timecode FindStartFromTakes(Vegas vegas, List<TakeSeg> takes, out string detail)
    {
        var pieces = new List<TimelinePiece>();
        foreach (Track t in vegas.Project.Tracks)
        {
            if (!t.IsAudio()) continue;
            foreach (TrackEvent ev in t.Events)
            {
                Take take = ev.ActiveTake;
                if (take == null || take.Media == null) continue;
                TimelinePiece p = new TimelinePiece();
                p.File = Path.GetFileName(take.Media.FilePath);
                p.StartMs = ev.Start.ToMilliseconds();
                p.OffsetMs = take.Offset.ToMilliseconds();
                p.LengthMs = ev.Length.ToMilliseconds();
                pieces.Add(p);
            }
        }
        double zeroMs = AnchorFromTakes(takes, pieces, out detail);
        return double.IsNaN(zeroMs) ? null : Timecode.FromMilliseconds(zeroMs);
    }

    // --- take anchor (no Vegas types: tested outside Vegas) ----------------

    public struct TimelinePiece
    {
        public string File;
        public double StartMs, OffsetMs, LengthMs;
    }

    // The take segment of the piece's file whose take range the piece
    // overlaps most, or -1 when the file is not listed or nothing overlaps.
    public static int MatchTake(List<TakeSeg> takes, TimelinePiece piece)
    {
        string file = piece.File.ToLowerInvariant();
        int best = -1;
        double bestOverlap = 0.0;
        for (int i = 0; i < takes.Count; i++)
        {
            if (takes[i].File.ToLowerInvariant() != file) continue;
            double overlap = Math.Min(piece.OffsetMs + piece.LengthMs, takes[i].ToMs)
                             - Math.Max(piece.OffsetMs, takes[i].FromMs);
            if (overlap > bestOverlap) { bestOverlap = overlap; best = i; }
        }
        return best;
    }

    // Narration zero on the timeline in ms, NaN when no piece matches. Every
    // matched piece implies event start - take offset - lag; the median wins.
    public static double AnchorFromTakes(List<TakeSeg> takes, List<TimelinePiece> pieces,
                                         out string detail)
    {
        var zeros = new List<double>();
        var lines = new List<string>();
        foreach (TimelinePiece p in pieces)
        {
            int i = MatchTake(takes, p);
            if (i < 0) continue;
            double implied = p.StartMs - p.OffsetMs - takes[i].LagMs;
            zeros.Add(implied);
            lines.Add("  " + takes[i].File + " (take " + (p.OffsetMs / 1000.0).ToString("0.0", CI) +
                      "s) at " + (p.StartMs / 1000.0).ToString("0.000", CI) +
                      "s -> narration zero " + (implied / 1000.0).ToString("0.000", CI) + "s");
        }
        if (zeros.Count == 0)
        {
            detail = "  none of the layout's take files is on the timeline";
            return double.NaN;
        }
        var sorted = new List<double>(zeros);
        sorted.Sort();
        double median = sorted[sorted.Count / 2];
        int off = 0;
        foreach (double z in zeros)
            if (Math.Abs(z - median) > AGREE_MS) off++;
        detail = string.Join("\n", lines.ToArray()) +
                 (off == 0 ? "\n  all " + zeros.Count + " take pieces agree"
                           : "\n  WARNING: " + off + " of " + zeros.Count +
                             " take pieces were moved after the mix was rendered - " +
                             "B-roll follows the majority");
        return median;
    }

    void RemoveTracksNamed(Vegas vegas, string name)
    {
        var doomed = new List<Track>();
        foreach (Track t in vegas.Project.Tracks)
            if (!t.IsAudio() && t.Name == name) doomed.Add(t);
        foreach (Track t in doomed)
            vegas.Project.Tracks.Remove(t);
    }

    // --- layout file ------------------------------------------------------

    string PickLayoutFile()
    {
        if (LAYOUT_FILE.Length > 0)
        {
            if (File.Exists(LAYOUT_FILE)) return LAYOUT_FILE;
            MessageBox.Show("Pinned layout not found:\n" + LAYOUT_FILE +
                            "\n\nFalling back to the file dialog.");
        }
        var dlg = new OpenFileDialog();
        dlg.Title = "Pick the project's layout.txt";
        dlg.Filter = "Clipmaker layout (layout*.txt)|layout*.txt|All files (*.*)|*.*";
        if (Directory.Exists(LAYOUTS_DIR)) dlg.InitialDirectory = LAYOUTS_DIR;
        return dlg.ShowDialog() == DialogResult.OK ? dlg.FileName : null;
    }

    public struct Ev
    {
        public long StartMs, LengthMs, TakeOffsetMs;
        public string Label, Path;
    }

    // One take=<file>|<from_ms>|<to_ms>|<lag_ms> header line: take time in
    // [FromMs, ToMs) sits in the narration at take time + LagMs.
    public struct TakeSeg
    {
        public string File;
        public long FromMs, ToMs, LagMs;
    }

    public class Layout
    {
        public string Project = "", Audio = "", Track = "B-Roll (layout)";
        public double Fps = 0, FadeMs = 0;
        public List<Ev> Events = new List<Ev>();
        public List<TakeSeg> Takes = new List<TakeSeg>();
    }

    public static TakeSeg ParseTake(string val)
    {
        string[] p = val.Split('|');
        if (p.Length != 4)
            throw new FormatException("Bad take line: " + val);
        TakeSeg t = new TakeSeg();
        t.File = p[0];
        t.FromMs = long.Parse(p[1], CI);
        t.ToMs = long.Parse(p[2], CI);
        t.LagMs = long.Parse(p[3], CI);
        if (t.ToMs <= t.FromMs)
            throw new FormatException("Empty take range: " + val);
        return t;
    }

    public static Layout ParseLayout(string path)
    {
        var layout = new Layout();
        bool inEvents = false;
        foreach (string raw in File.ReadAllLines(path))
        {
            string line = raw.Trim();
            if (line.Length == 0 || line.StartsWith("#")) continue;
            if (line == "---") { inEvents = true; continue; }

            if (!inEvents)
            {
                int eq = line.IndexOf('=');
                if (eq < 0) continue;
                string key = line.Substring(0, eq);
                string val = line.Substring(eq + 1);
                if (key == "project") layout.Project = val;
                else if (key == "audio") layout.Audio = val;
                else if (key == "track") layout.Track = val;
                else if (key == "fps") layout.Fps = double.Parse(val, CI);
                else if (key == "fade_ms") layout.FadeMs = double.Parse(val, CI);
                else if (key == "take") layout.Takes.Add(ParseTake(val));
                continue;
            }

            string[] parts = line.Split(new char[] { '|' }, 5);
            if (parts.Length < 5)
                throw new FormatException("Bad layout row: " + line);
            Ev e = new Ev();
            e.StartMs = long.Parse(parts[0], CI);
            e.LengthMs = long.Parse(parts[1], CI);
            e.TakeOffsetMs = long.Parse(parts[2], CI);
            e.Label = parts[3];
            e.Path = parts[4];
            layout.Events.Add(e);
        }
        return layout;
    }
}
