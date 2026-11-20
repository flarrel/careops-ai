# From lookup to review

The [12-second explainer](../creative/revised.mp4) shows how CareOps AI presents a cited policy answer, abstains without sufficient evidence, and pauses a synthetic fixture lookup for human approval. Its captions describe software behavior. The illustration uses abstract documents and a fictional reviewer.

| Time | Storyboard action | Behavior reference |
| --- | --- | --- |
| 0–4 s | A marker connects the source-page highlight to the excerpt. | [Grounded policy answers](grounded-answers.md) |
| 4–8 s | A broken evidence route ends in an amber abstention card. | [Grounded policy answers](grounded-answers.md) |
| 8–12 s | The reviewer nods and presses Allow lookup. A marker crosses to the fixture and reveals a read-only lookup. | [Review agent](review-agent.md) |

## Build and revise

The [storyboard](../creative/storyboard.json) holds the shot order, captions, and visual choices. The [scene generator](../creative/scene.py) builds the geometry, camera cuts, and keyframed character and graphics. Open the [editable revised scene](../creative/revised.blend) in Blender and scrub frames 1–144. Camera cuts are at frames 1, 49, and 97. The approval pose is at frame 119.

With Blender 4.5 and FFmpeg with libx264 available, run from the repository root. The build script writes to the output path and uses temporary space for the frame sequences.

```sh
python3 creative/build.py /tmp/careops-explainer
```

Use `--blender` and `--ffmpeg` for binaries outside the shell path. The render runs Cycles on CPU. FFmpeg assembles the PNGs at 12 fps, applies four-frame opening and closing fades, exports a browser-compatible H.264 MP4, and decodes the full result to check the frame count. [Blender command-line rendering](https://docs.blender.org/manual/en/4.5/advanced/command_line/render.html), [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html)

## Prompt experiments

A local Qwen2.5-0.5B-Instruct run ranked three candidate visual beats for each shot. The initial prompt gave a general creative brief, the revised prompt emphasized observable cause and effect, the checklist added software constraints, and the action prompt prioritized visible verbs. The prompts agreed with the source-backed editorial choices on 2/3, 2/3, 1/3, and 2/3 shots. None of the rewrites improved agreement on this three-shot set. The first two prompts selected the source-page tab and the broken route. Their payment-approval suggestion failed the approval boundary, so the scene uses the reviewer’s action and read-only result.

Replay all twelve decisions with the local model from the [answer service guide](grounded-answers.md). The [prompt log](../creative/prompt-results.json) preserves each exact system prompt, input, constrained response, and scoring choice.

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=src python3 creative/prompts.py \
  models/qwen2.5-0.5b-instruct /tmp/careops-prompts.json
```

Local text inference gave fast storyboard suggestions. Blender keeps the captions and order of operations directly editable. Wan2.1 text-to-video was reviewed as an option for generated motion. Its authors report 8.19 GB of VRAM for its small model and about four minutes for a five-second 480p clip on an RTX 4090. That is an upstream measurement on different hardware. This workflow used the existing local text model and keyframed graphics to preserve exact labels and action timing. It did not benchmark video generation or measure a human-only storyboard workflow. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Wan2.1 release and benchmarks](https://github.com/Wan-Video/Wan2.1)

## Revision review

The [baseline](../creative/baseline.mp4) and revised cut use the same three four-second shots. The [frame comparison](../creative/revision-review.jpg) shows baseline above revised. The revision adds a readable source-page tab, delays and colors the abstention card so the broken route registers first, and separates the approval pose from the lookup with a visible traveling marker.

The [build record](../creative/build-results.json) reports scene construction, CPU rendering, editing and encoding wall time, file size, and decoded frame count for both cuts. The timings come from one local run per version and vary by hardware and machine load. Frame review checked the citation, abstention, and approval sequence. The [approval sequence](../creative/approval-sequence.jpg) shows the fixture panel before the press, at the press, and after the lookup appears.
