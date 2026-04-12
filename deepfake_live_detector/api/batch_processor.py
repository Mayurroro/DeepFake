"""Batch processor — sends many files to the FastAPI /detect endpoint concurrently."""
import os, glob, json, time, argparse, requests
from concurrent.futures import ThreadPoolExecutor

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "heic", "bmp", "tiff"}
AUDIO_EXTS = {"mp3", "wav", "m4a", "flac", "ogg"}
ALL_EXTS = IMAGE_EXTS | AUDIO_EXTS


def process_file(file_path, api_url):
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(api_url, files={"file": (os.path.basename(file_path), f)})
            if resp.status_code == 200:
                return {**resp.json(), "status": "success"}
            return {"filename": os.path.basename(file_path), "status": "failed", "error": resp.text}
    except Exception as e:
        return {"filename": os.path.basename(file_path), "status": "failed", "error": str(e)}


def run_batch(directory, output="batch_results.json", api_url="http://localhost:8000/detect", workers=10):
    files = []
    for ext in ALL_EXTS:
        files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
        files.extend(glob.glob(os.path.join(directory, f"*.{ext.upper()}")))
    print(f"Found {len(files)} files in {directory}. Workers: {workers}")

    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(process_file, f, api_url) for f in files]
        for i, fut in enumerate(futs):
            results.append(fut.result())
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(files)} done")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Batch done in {time.time()-t0:.1f}s → {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch deepfake detection via API")
    p.add_argument("--dir", required=True, help="Directory of files to scan")
    p.add_argument("--out", default="batch_results.json")
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    run_batch(a.dir, a.out, workers=a.workers)
