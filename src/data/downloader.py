"""Data download and local caching module."""
import os
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from src.utils.config import get_project_root, resolve_path, setup_logger

logger = setup_logger("downloader")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def validate_download_url(url: str):
    """Validate that the URL has an allowed safe scheme."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsafe URL scheme '{parsed.scheme}'. Only http and https are allowed.")

def download_data(
    url: str = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv",
    output_path: Optional[str] = None,
    max_lines: Optional[int] = 300000,
    force_download: bool = False,
    timeout_seconds: float = 30.0
) -> Path:
    """
    Download raw Twitter Customer Support dataset.
    If max_lines is specified, streams only the first N lines to save bandwidth and speed up local execution.
    """
    validate_download_url(url)
    
    if output_path is None:
        target_file = resolve_path("data/raw/twcs_sample.csv")
    else:
        target_file = resolve_path(output_path)
    
    target_file.parent.mkdir(parents=True, exist_ok=True)
    
    if target_file.exists() and not force_download:
        logger.info(f"Using cached dataset at: {target_file}")
        return target_file

    logger.info(f"Downloading dataset from: {url}")
    logger.info(f"Target local file: {target_file}")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    if max_lines:
        logger.info(f"Streaming first {max_lines:,} lines for fast reproducible execution...")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp, open(target_file, "w", encoding="utf-8", errors="replace") as out_f:
            count = 0
            pbar = tqdm(total=max_lines, desc="Streaming lines", unit="lines")
            for line in resp:
                decoded = line.decode("utf-8", errors="ignore")
                out_f.write(decoded)
                count += 1
                pbar.update(1)
                if count >= max_lines:
                    break
            pbar.close()
    else:
        # Stream full download with timeout
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp, open(target_file, "wb") as out_f:
            total_size = int(resp.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading full dataset") as pbar:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    pbar.update(len(chunk))
            
    logger.info(f"Dataset successfully saved to: {target_file}")
    return target_file
