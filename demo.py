"""Local Gradio demo containing every real panorama from examples/."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"


def sample_images():
    from files_inference import collect_images
    return collect_images(EXAMPLES)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/pytorch_model.bin")
    parser.add_argument("--sky-model", type=Path, default=ROOT / "checkpoints/skyseg.onnx")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def build_demo(checkpoint, sky_model=ROOT / "checkpoints/skyseg.onnx", device="cuda", precision="auto"):
    import gradio as gr
    from PIL import Image

    checkpoint = Path(checkpoint).expanduser().resolve()
    output_root = ROOT / "outputs" / "demo_ui"
    input_root = output_root / "inputs"
    examples = sample_images()

    def reconstruct(image_path, remove_sky):
        if not image_path:
            raise gr.Error("Choose a real example or upload a 2:1 panorama.")
        if not checkpoint.is_file():
            raise gr.Error(f"GeoPano checkpoint not found: {checkpoint}")
        if remove_sky and not Path(sky_model).is_file():
            raise gr.Error("Sky model missing. Supply --sky-model /path/to/skyseg.onnx or disable Remove sky points.")
        source = Path(image_path)
        with Image.open(source) as image:
            width, height = image.size
            if abs(width / height - 2) > 0.02:
                raise gr.Error(f"Expected a 2:1 panorama; received {width} x {height}.")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            input_root.mkdir(parents=True, exist_ok=True)
            staged = input_root / f"panorama_{digest}.png"
            if not staged.exists():
                image.convert("RGB").save(staged)
        command = [sys.executable, str(ROOT / "files_inference.py"),
                   "--input", str(staged), "--checkpoint", str(checkpoint),
                   "--output", str(output_root), "--device", device, "--precision", precision, "--overwrite"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise gr.Error((result.stderr or result.stdout)[-3000:])
        out = output_root / staged.stem
        ply = out / "pano_points.ply"
        if remove_sky:
            from geopano.sky_filter import export_without_sky
            try:
                ply = export_without_sky(staged, out, sky_model)
            except Exception as error:
                raise gr.Error(f"Sky filtering failed: {error}") from error
        return str(ply), str(ply)

    with gr.Blocks(title="GeoPano — Real panorama demo") as demo:
        gr.Markdown("# GeoPano — real panorama demo. Select any bundled real panorama or upload another 2:1 RGB panorama. Each image is reconstructed independently.")
        panorama = gr.Image(label="Equirectangular panorama", type="filepath")
        gr.Examples(examples=[[str(path)] for path in examples], inputs=[panorama], label="Real sample panoramas")
        remove_sky = gr.Checkbox(value=Path(sky_model).is_file(), label="Remove sky points")
        run = gr.Button("Run GeoPano", variant="primary")
        with gr.Row():
            cloud = gr.Model3D(label="Colored point cloud", display_mode="point_cloud")
        download = gr.File(label="Download colored PLY")
        run.click(reconstruct, inputs=[panorama, remove_sky], outputs=[cloud, download], concurrency_limit=1)
    return demo


def main():
    args = parse_args()
    demo = build_demo(args.checkpoint, args.sky_model, args.device, args.precision)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port,
        allowed_paths=[str(ROOT / "outputs")], share=False)


if __name__ == "__main__":
    main()
