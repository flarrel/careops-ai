"""Compare storyboard prompts with the same local model and choice set."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from careops.answer import LocalGenerator, prompt_data

PROMPTS = {
    "initial": "You are a film storyboard artist. Choose the strongest visual for this shot.",
    "revised": (
        "You are directing a silent four-second product explainer. Choose the visual "
        "that makes the behavior in the subtitle observable. Use a concrete prop or "
        "action that shows the stated cause and effect. Do not add predictions, "
        "payment authority, or an outcome that contradicts the subtitle. Return its letter."
    ),
    "source_checklist": (
        "Pick a letter for a four-second CareOps AI storyboard. A policy answer "
        "must show its source page. Missing evidence must show abstention. Human "
        "approval opens only a read-only synthetic fixture lookup, after the "
        "reviewer acts. A badge that says payment approved is wrong because "
        "this workflow does not approve payments. Choose the option that "
        "demonstrates the matching behavior. Return only the letter."
    ),
    "visual_action": (
        "Choose a storyboard direction for the supplied subtitle. Prioritize the "
        "option that shows the actor performing the subtitle's verb on screen. "
        "Information shown in the shot cannot change the result of the software's "
        "action. A badge that shows a different outcome is always a worse fit. "
        "Answer with one option letter."
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    storyboard = json.loads(Path(__file__).with_name("storyboard.json").read_text())
    model = LocalGenerator(args.model)
    experiments = {}
    for name, system in PROMPTS.items():
        decisions = []
        for shot in storyboard["shots"]:
            message = prompt_data(
                shot["title"], subtitle=shot["subtitle"], options=shot["focal_options"]
            )
            start = perf_counter()
            choice = model.choose(system, message, tuple(shot["focal_options"]))
            decisions.append({
                "message": message,
                "choice": choice,
                "seconds": round(perf_counter() - start, 3),
                "editor_choice": shot["focal_choice"],
            })
        experiments[name] = {
            "system": system,
            "decisions": decisions,
            "editor_agreements": sum(item["choice"] == item["editor_choice"] for item in decisions),
        }
    report = {"model": "Qwen2.5-0.5B-Instruct", "experiments": experiments}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
