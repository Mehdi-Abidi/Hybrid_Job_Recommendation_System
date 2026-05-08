import multiprocessing
import time
from huggingface_hub import snapshot_download
import sys

def download():
    snapshot_download("sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":
    success = False
    for i in range(250):
        print(f"Attempt {i}...", flush=True)
        p = multiprocessing.Process(target=download)
        p.start()
        p.join(15) # Wait exactly 15 seconds
        if p.is_alive():
            print("Timed out! Network hung. Killing process...", flush=True)
            p.terminate()
            p.join()
        elif p.exitcode == 0:
            print("HF Success!", flush=True)
            success = True
            break
        else:
            print(f"Process crashed with exit code {p.exitcode}. Retrying...", flush=True)
            
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
