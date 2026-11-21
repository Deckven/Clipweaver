// Test harness for vegas_scripts/import_layout.cs: runs the loader's own
// ParseLayout / AnchorFromTakes outside Vegas (they use no Vegas types).
// Compiled and driven by tests/test_vegas_loader.py.
//
//   LayoutHarness parse  <layout.txt>
//   LayoutHarness anchor <layout.txt> <pieces.txt>   (lines: file|start_ms|offset_ms|length_ms)
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

public static class LayoutHarness
{
    static readonly CultureInfo CI = CultureInfo.InvariantCulture;

    public static int Main(string[] args)
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        EntryPoint.Layout layout = EntryPoint.ParseLayout(args[1]);
        if (args[0] == "parse")
        {
            Console.WriteLine("header|" + layout.Project + "|" + layout.Audio + "|" + layout.Track +
                              "|" + layout.Fps.ToString("R", CI) + "|" + layout.FadeMs.ToString("R", CI));
            foreach (EntryPoint.TakeSeg t in layout.Takes)
                Console.WriteLine("take|" + t.File + "|" + t.FromMs + "|" + t.ToMs + "|" + t.LagMs);
            foreach (EntryPoint.Ev e in layout.Events)
                Console.WriteLine("ev|" + e.StartMs + "|" + e.LengthMs + "|" + e.TakeOffsetMs +
                                  "|" + e.Label + "|" + e.Path);
            return 0;
        }
        if (args[0] == "anchor")
        {
            var pieces = new List<EntryPoint.TimelinePiece>();
            foreach (string line in File.ReadAllLines(args[2]))
            {
                if (line.Trim().Length == 0) continue;
                string[] p = line.Split('|');
                EntryPoint.TimelinePiece piece = new EntryPoint.TimelinePiece();
                piece.File = p[0];
                piece.StartMs = double.Parse(p[1], CI);
                piece.OffsetMs = double.Parse(p[2], CI);
                piece.LengthMs = double.Parse(p[3], CI);
                pieces.Add(piece);
            }
            string detail;
            double zero = EntryPoint.AnchorFromTakes(layout.Takes, pieces, out detail);
            Console.WriteLine("zero|" + (double.IsNaN(zero) ? "NaN" : zero.ToString("R", CI)));
            Console.WriteLine(detail);
            return 0;
        }
        Console.Error.WriteLine("usage: LayoutHarness parse|anchor <layout.txt> [pieces.txt]");
        return 2;
    }
}
