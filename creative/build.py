"""Render both versions, edit an MP4 from each image sequence, and verify decoding."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter


HERE = Path(__file__).resolve().parent
VERSIONS = ("baseline", "revised")


def capture(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def review_images(ffmpeg, output):
    shots = "select='eq(n,37)+eq(n,85)+eq(n,138)',scale=480:270,tile=3x1"
    approval = "select='eq(n,103)+eq(n,118)+eq(n,138)',scale=480:270,tile=3x1"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    image = ["-frames:v", "1", "-q:v", "2"]
    subprocess.run([
        *command, "-i", output / "baseline.mp4", "-i", output / "revised.mp4",
        "-filter_complex", f"[0:v]{shots}[before];[1:v]{shots}[after];[before][after]vstack[out]",
        "-map", "[out]", *image, output / "revision-review.jpg",
    ], check=True)
    subprocess.run([
        *command, "-i", output / "revised.mp4", "-vf", approval,
        *image, output / "approval-sequence.jpg",
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((HERE / "storyboard.json").read_text())
    fps = plan["fps"]
    frame_count = fps * plan["seconds_per_shot"] * len(plan["shots"])
    filters = f"setsar=1,fade=in:s=0:n=4,fade=out:s={frame_count - 5}:n=4"
    report = {
        "blender": next(line for line in capture([args.blender, "--version"]).splitlines()
                        if line.startswith("Blender ")),
        "ffmpeg": capture([args.ffmpeg, "-version"]).splitlines()[0],
        "fps": fps, "dimensions": [960, 540], "expected_frames": frame_count,
        "edit_filters": filters,
        "versions": {},
    }
    for version in VERSIONS:
        with TemporaryDirectory(prefix=f"careops-{version}-") as temporary:
            frames = Path(temporary)
            blend = output / f"{version}.blend"
            mp4 = output / f"{version}.mp4"
            start = perf_counter()
            with (output / f"{version}.log").open("w") as log:
                subprocess.run([
                    args.blender, "-b", "--python-exit-code", "1", "--python",
                    HERE / "scene.py", "--", version, blend,
                ], stdout=log, stderr=subprocess.STDOUT, check=True)
                constructed = perf_counter()
                subprocess.run([
                    args.blender, "-b", blend, "-o", frames / "frame", "-a",
                ], stdout=log, stderr=subprocess.STDOUT, check=True)
            rendered = perf_counter()
            paths = sorted(frames.glob("frame*.png"))
            expected = [frames / f"frame{index:04d}.png" for index in range(1, frame_count + 1)]
            if paths != expected:
                raise RuntimeError(f"Incomplete frame sequence for {version}")
            subprocess.run([
                args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(fps), "-i", frames / "frame%04d.png",
                "-vf", filters, "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", mp4,
            ], check=True)
            edited = perf_counter()
            progress = capture([
                args.ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror",
                "-progress", "pipe:1", "-i", mp4, "-f", "null", "-",
            ])
            decoded = [int(line.split("=")[1]) for line in progress.splitlines()
                       if line.startswith("frame=")][-1]
            if decoded != frame_count:
                raise RuntimeError(f"{version} decoded {decoded} frames, expected {frame_count}")
            report["versions"][version] = {
                "scene_seconds": round(constructed - start, 2),
                "render_seconds": round(rendered - constructed, 2),
                "edit_seconds": round(edited - rendered, 2),
                "decoded_frames": decoded,
                "bytes": mp4.stat().st_size,
            }
    review_images(args.ffmpeg, output)
    (output / "build-results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
