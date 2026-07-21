"""Verify the SO-101 environment and probe signal in simulation.

This script loads the MuJoCo environment for both shells, renders a camera
frame, runs the scripted probe, and prints the extracted signal features.
"""

from pathlib import Path

from PIL import Image

from probenet.env import SO101Env
from probenet.probe import ProbeRunner, default_probe_config, extract_probe_features


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = default_probe_config()

    results: dict[str, dict] = {}
    for object_type in ("shell_a", "shell_b"):
        env = SO101Env(object_type=object_type, image_size=(224, 224))
        env.reset(seed=0)

        rgb = env.render()
        img_path = output_dir / f"probe_verify_{object_type}.png"
        Image.fromarray(rgb).save(img_path)

        runner = ProbeRunner(env, config)
        signal = runner.run()
        features = extract_probe_features(signal)
        results[object_type] = {
            "image_path": str(img_path),
            "features": features,
            "timesteps": len(signal["timestamps"]),
        }
        env.close()

    print("\n=== Probe verification results ===\n")
    for object_type, data in results.items():
        print(f"{object_type}: {data['timesteps']} probe steps")
        print(f"  image: {data['image_path']}")
        for name, value in data["features"].items():
            print(f"  {name}: {value:.6f}")
        print()

    af_a = results["shell_a"]["features"]["af_mean"]
    af_b = results["shell_b"]["features"]["af_mean"]
    ratio = abs(af_b - af_a) / (abs(af_a) + 1e-8)
    print(f"Mean actuator force contrast: {ratio:.2%}")

    if ratio > 0.1:
        print("Probe signal distinguishes the two shells.")
    else:
        print("WARNING: probe signal does not clearly distinguish the shells.")


if __name__ == "__main__":
    main()
