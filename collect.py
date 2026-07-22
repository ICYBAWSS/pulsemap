#!/usr/bin/env python3
"""
PulseMap sample collector.

Paste a Google Drive link, press enter, and this script:
  1. downloads it (gdown)          -- a single .zip file OR a shared folder
  2. unzips any archives
  3. auto-sorts every drum ONE-SHOT:
       - a SPECIFIC type (Kick, Snare, Rimshot, Shaker, ...) -> training_data/labeled/<Type>/
       - a GRAB-BAG folder (Percs, One Shots, Misc, FX, ...) -> training_data/unsorted/
     using labels.py. Grab-bag folders are deliberately NOT used as training
     labels -- see the big comment in labels.py for why ("Percs" mixes rimshot +
     shaker + tambourine + cowbell into one muddy blob and contradicts sounds
     that got a specific label elsewhere). Unsorted files still count for the
     visual map later; they just don't teach the classifier.
  4. de-dupes (kits repost the same samples constantly)
  5. prints a live tally of how close each SPECIFIC category is to its target

Keep pasting links until the bars fill up. Labeled one-shots land in
./training_data/labeled/<Type>/, grab-bag leftovers in ./training_data/unsorted/.

Commands at the prompt:
  <a google drive link>   process it
  status                  reprint the tally
  target <N>              set the per-category goal (default 150)
  open                    print the labeled folder path
  quit / q                exit

Run it:  source venv-clap/bin/activate && python collect.py
"""
import os
import sys
import csv
import shutil
import hashlib
import zipfile
import tempfile
import subprocess
import queue
import threading
import time
import re
import requests
import traceback
from pathlib import Path

import soundfile as sf

from labels import classify_path, LABELS, UNSORTED

# ---- config -----------------------------------------------------------------
DATA_DIR = Path("training_data")
RAW_DIR = DATA_DIR / "raw"          # downloaded archives / folders (gitignore)
LABELED_DIR = DATA_DIR / "labeled"  # the payoff: specific-type one-shots (trainable)
UNSORTED_DIR = DATA_DIR / "unsorted"  # grab-bag leftovers (map-only, not trainable)
MANIFEST = DATA_DIR / "manifest.csv"
SEEN_FILE = DATA_DIR / "seen_hashes.txt"
LINKS_FILE = DATA_DIR / "processed_links.txt"
LOG_FILE = DATA_DIR / "collect.log"

TARGET_PER_CLASS = 150
MAX_ONESHOT_SEC = 3.0               # anything longer is treated as a loop and skipped
AUDIO_EXTS = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")
ARCHIVE_EXTS = (".zip",)            # .rar needs `brew install unar` (see extract())

# ---- tiny helpers -----------------------------------------------------------
def log_to_file(msg):
    """Write message to a log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass


def open_log_terminal():
    """Launch a new macOS Terminal window tailing the log file."""
    log_path = LOG_FILE.resolve()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    
    cmd = [
        "osascript", "-e",
        f'tell application "Terminal" to do script '
        f'"clear && echo \\"\033[1;36m=== PulseMap Background Worker Logs ===\033[0m\\" && '
        f'echo \\"Tailing log file at: {log_path}\\" && echo \\"\\" && '
        f'tail -n 100 -f \'{log_path}\'"'
    ]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def print_tally_to_log(counts, target):
    """Writes the tally progress bar chart to the log file."""
    lines = []
    lines.append(f"Progress  (target {target}/category · specific types only)")
    done = 0
    for lbl in sorted(LABELS, key=lambda k: counts[k]):
        n = counts[lbl]
        frac = min(n / target, 1.0) if target else 0
        filled = int(frac * 22)
        bar = "█" * filled + "░" * (22 - filled)
        if n >= target:
            done += 1
            mark, col = "✓", C_GREEN
        elif n == 0:
            mark, col = " ", C_DIM
        else:
            mark, col = " ", C_YEL
        lines.append(f"  {col}{lbl:11s} {n:4d}/{target:<4d} [{bar}] {mark}{C_RST}")
    total = sum(counts.values())
    unsorted_n = unsorted_count()
    lines.append(f"\n  {C_BOLD}{total} labeled one-shots · {done}/{len(LABELS)} categories at target{C_RST}")
    lines.append(f"  {C_DIM}+ {unsorted_n} in unsorted/ (grab-bag folders — Percs, Misc, FX, ... — held out of training, used for the map){C_RST}")
    
    for line in lines:
        log_to_file(line)


def clean_local_path(raw_path):
    """Cleans up paths pasted or dragged-and-dropped into terminal (handles quotes and backslash-escaped spaces)."""
    path_str = raw_path.strip()
    if (path_str.startswith("'") and path_str.endswith("'")) or (path_str.startswith('"') and path_str.endswith('"')):
        path_str = path_str[1:-1].strip()
    path_str = path_str.replace("\\ ", " ")
    return path_str


# ---- tiny helpers -----------------------------------------------------------
C_GREEN, C_YEL, C_DIM, C_RST, C_BOLD = "\033[32m", "\033[33m", "\033[2m", "\033[0m", "\033[1m"


def sh(msg):  # section header
    print(f"\n{C_BOLD}{msg}{C_RST}")


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def duration_of(path):
    """Fast duration read without decoding the whole file."""
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:
        return None


# ---- state ------------------------------------------------------------------
link_queue = queue.Queue()
current_link = None
current_link_lock = threading.Lock()


def load_seen():
    if SEEN_FILE.exists():
        return set(SEEN_FILE.read_text().split())
    return set()


def append_seen(h):
    with open(SEEN_FILE, "a") as f:
        f.write(h + "\n")


def append_manifest(row):
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["hash", "label", "duration", "orig_name", "source_folder", "source_link"])
        w.writerow(row)


def tally():
    """Counts for TRAINABLE (specific-type) classes only -- excludes unsorted."""
    counts = {lbl: 0 for lbl in LABELS}
    for lbl in LABELS:
        d = LABELED_DIR / lbl
        if d.exists():
            counts[lbl] = sum(1 for _ in d.glob("*") if _.is_file())
    return counts


def unsorted_count():
    return sum(1 for _ in UNSORTED_DIR.glob("*") if _.is_file()) if UNSORTED_DIR.exists() else 0


# ---- download ---------------------------------------------------------------
def extract_mediafire_link(html):
    # Try using bs4 first
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try id downloadButton
        btn = soup.find('a', id='downloadButton')
        if btn and btn.get('href'):
            return btn.get('href').strip()
            
        # Try any link in class 'download_link'
        btn = soup.select_one('.download_link a')
        if btn and btn.get('href'):
            return btn.get('href').strip()
            
        # Try any link containing 'download' class
        btn = soup.select_one('a[href*="download"]')
        if btn and btn.get('href'):
            return btn.get('href').strip()
    except Exception:
        pass
        
    # Fallback to regex
    match = re.search(r'href="((https?://download[^"]+))"', html)
    if match:
        return match.group(1).strip()
    
    # Try searching for download link pattern directly without href prefix
    match = re.search(r'https?://download[^\'\"]+', html)
    if match:
        return match.group(0).strip()
        
    return None


def download_mediafire(link, dest_dir):
    """Downloads a file from MediaFire into dest_dir/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        res = sess.get(link, timeout=60)
        res.raise_for_status()
    except Exception as e:
        log_to_file(f"  failed to connect to MediaFire link: {e}")
        return False
        
    direct_url = extract_mediafire_link(res.text)
    if not direct_url:
        log_to_file("  could not find direct download link on the MediaFire page.")
        return False
        
    log_to_file(f"Found direct download link: {direct_url}")
    
    # Extract filename
    from urllib.parse import urlparse, unquote
    filename = None
    if 'Content-Disposition' in res.headers:
        cd_match = re.search(r'filename="?([^";]+)"?', res.headers['Content-Disposition'])
        if cd_match:
            filename = cd_match.group(1)
            
    if not filename:
        parsed_url = urlparse(direct_url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "downloaded_file.zip"
            
    filename = unquote(filename)
    dest_file = dest_dir / filename
    
    # Method 1: requests download
    log_to_file(f"Downloading {filename} via requests...")
    try:
        with sess.get(direct_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            log_to_file(f"  download complete: {filename}")
            return True
    except Exception as e:
        log_to_file(f"  requests download failed: {e}. Retrying with curl...")
        
    # Method 2: curl download
    cmd = ["curl", "-L", "-o", str(dest_file), direct_url]
    log_to_file(f"Method (curl): Running curl copy: {' '.join(cmd)}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_f:
            subprocess.run(cmd, stdout=log_f, stderr=log_f, timeout=1200)
        if dest_file.exists() and dest_file.stat().st_size > 0:
            log_to_file(f"  curl download complete: {filename}")
            return True
    except Exception as e:
        log_to_file(f"  curl download failed: {e}")
        
    return False


def load_cookies_txt(file_path):
    """Loads a Netscape cookies.txt file into a dictionary for requests."""
    cookies = {}
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 7:
                    domain = parts[0]
                    name = parts[5]
                    value = parts[6].strip()
                    if "google" in domain:
                        cookies[name] = value
    except Exception as e:
        log_to_file(f"  Error loading cookies.txt: {e}")
    return cookies


def get_rclone_drive_remote():
    """Checks if there is a Google Drive remote configured in rclone."""
    try:
        r = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if ":" in line:
                    name = line.split(":")[0].strip()
                    r_show = subprocess.run(["rclone", "config", "show", name], capture_output=True, text=True, timeout=10)
                    if "type = drive" in r_show.stdout:
                        return name
    except Exception:
        pass
    return None


def run_rclone_cmd(remote_name, gdrive_id, dest):
    """Downloads a file or folder from Google Drive using rclone config."""
    cmd = [
        "rclone", "copy",
        f"{remote_name},root_folder_id={gdrive_id}:",
        str(dest)
    ]
    log_to_file(f"Method (rclone): Running rclone copy from remote '{remote_name}' for ID {gdrive_id}")
    log_to_file(f"$ {' '.join(cmd)}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_f:
            log_f.write(f"--- rclone start: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_f.flush()
            r = subprocess.run(
                cmd,
                cwd=dest,
                timeout=1800,
                stdout=log_f,
                stderr=log_f
            )
            log_f.write(f"--- rclone end: {time.strftime('%Y-%m-%d %H:%M:%S')} (exit code: {r.returncode}) ---\n")
            log_f.flush()
        return r.returncode == 0
    except Exception as e:
        log_to_file(f"  rclone command failed: {e}")
        return False


def run_curl_download(file_id, dest_dir, cookies_path=None):
    """Downloads a public Google Drive file using curl, attempting to bypass the virus warning."""
    import tempfile
    
    cookies_arg = f"-b '{cookies_path}'" if cookies_path and cookies_path.exists() else ""
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    
    # Create temp files to store headers and warning page
    with tempfile.NamedTemporaryFile(suffix=".txt") as head_f, tempfile.NamedTemporaryFile(suffix=".html") as html_f:
        cmd_step1 = f"curl -s -L -D '{head_f.name}' {cookies_arg} '{url}' -o '{html_f.name}'"
        log_to_file(f"  Method (curl): Running Step 1: {cmd_step1}")
        
        r1 = subprocess.run(cmd_step1, shell=True)
        if r1.returncode != 0:
            return False
            
        confirm_token = None
        try:
            with open(html_f.name, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
                match = re.search(r'confirm=([a-zA-Z0-9_]+)', html_content)
                if match:
                    confirm_token = match.group(1)
        except Exception:
            pass
            
        filename = None
        try:
            with open(head_f.name, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "Content-Disposition" in line:
                        cd_match = re.search(r'filename="?([^";\r\n]+)"?', line)
                        if cd_match:
                            filename = cd_match.group(1).strip()
        except Exception:
            pass
            
        if not filename:
            filename = f"gdrive_curl_{file_id}.zip"
            
        if (filename.startswith('"') and filename.endswith('"')) or (filename.startswith("'") and filename.endswith("'")):
            filename = filename[1:-1]
            
        dest_file = dest_dir / filename
        
        download_url = url
        if confirm_token:
            log_to_file(f"  Method (curl): Bypassing warning with token: {confirm_token}")
            download_url += f"&confirm={confirm_token}"
            
        cmd_step2 = f"curl -L {cookies_arg} '{download_url}' -o '{dest_file}'"
        log_to_file(f"  Method (curl): Running Step 2: {cmd_step2}")
        
        r2 = subprocess.run(cmd_step2, shell=True)
        return r2.returncode == 0 and dest_file.exists() and dest_file.stat().st_size > 0


def download_gdrive_file_requests(file_id, dest_dir, cookies=None):
    """Downloads a public Google Drive file using requests, bypassing the virus warning page."""
    session = requests.Session()
    if cookies:
        session.cookies.update(cookies)
        
    url = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}
    
    log_to_file(f"  Attempting requests fallback download for file ID: {file_id}")
    try:
        response = session.get(url, params=params, stream=True, timeout=60)
        
        confirm_token = None
        for key, val in response.cookies.items():
            if key.startswith("download_warning"):
                confirm_token = val
                break
                
        if not confirm_token:
            html = response.text
            match = re.search(r'confirm=([a-zA-Z0-9_]+)', html)
            if match:
                confirm_token = match.group(1)
                
        if confirm_token:
            log_to_file(f"  Bypassing Google virus warning with token: {confirm_token}")
            params["confirm"] = confirm_token
            response = session.get(url, params=params, stream=True, timeout=1200)
            
        response.raise_for_status()
        
        filename = None
        if 'Content-Disposition' in response.headers:
            cd_match = re.search(r'filename="?([^";]+)"?', response.headers['Content-Disposition'])
            if cd_match:
                filename = cd_match.group(1)
        if not filename:
            filename = f"gdrive_file_{file_id}.zip"
            
        # Clean filename from potential quotes
        if (filename.startswith('"') and filename.endswith('"')) or (filename.startswith("'") and filename.endswith("'")):
            filename = filename[1:-1]
            
        dest_file = dest_dir / filename
        log_to_file(f"  Downloading {filename} via requests...")
        
        with open(dest_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        log_to_file(f"  requests download complete: {filename}")
        return True
    except Exception as e:
        log_to_file(f"  requests fallback failed: {e}")
        return False


def crawl_public_folder(folder_id):
    """Crawls a public Google Drive folder page and extracts all file IDs and names."""
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
    log_to_file(f"  Crawling folder view URL: {url}")
    try:
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        html = res.text
        
        entries = []
        matches = re.findall(r'\["([a-zA-Z0-9_-]{25,})","([^"]+)"', html)
        for fid, name in matches:
            if fid == folder_id:
                continue
            try:
                name = name.encode('utf-8').decode('unicode-escape')
            except Exception:
                pass
            entries.append((fid, name))
            
        seen_ids = set()
        unique_entries = []
        for fid, name in entries:
            if fid not in seen_ids:
                seen_ids.add(fid)
                unique_entries.append((fid, name))
                
        log_to_file(f"  Found {len(unique_entries)} entries in folder crawl.")
        return unique_entries
    except Exception as e:
        log_to_file(f"  Folder crawling failed: {e}")
        return []


def download_folder_crawled(folder_id, dest_dir, cookies=None):
    """Crawls and downloads all files in a public Google Drive folder recursively."""
    entries = crawl_public_folder(folder_id)
    if not entries:
        return False
        
    success_count = 0
    cookies_path = Path("cookies.txt")
    for fid, name in entries:
        log_to_file(f"  Crawled file entry: {name} ({fid})")
        if download_gdrive_file_requests(fid, dest_dir, cookies=cookies):
            success_count += 1
        elif run_curl_download(fid, dest_dir, cookies_path=cookies_path if cookies else None):
            success_count += 1
                
    log_to_file(f"  Crawled download complete. Successfully downloaded {success_count}/{len(entries)} files.")
    return success_count > 0


def get_gdrive_id(url):
    """Extract file/folder ID from a Google Drive URL."""
    # Pattern for folder
    m = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1), True
    # Pattern for open?id=
    m = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if m:
        is_folder = "folder" in url.lower()
        return m.group(1), is_folder
    # Pattern for file/d/
    m = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1), False
    return None, False


def run_gdown_cmd(cmd, dest):
    """Helper to run a gdown command, writing output directly to the log file."""
    log_to_file(f"$ {' '.join(cmd)}")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_f:
            log_f.write(f"--- gdown start: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_f.flush()
            r = subprocess.run(
                cmd,
                cwd=dest,
                timeout=1800,
                stdout=log_f,
                stderr=log_f
            )
            log_f.write(f"--- gdown end: {time.strftime('%Y-%m-%d %H:%M:%S')} (exit code: {r.returncode}) ---\n")
            log_f.flush()
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        log_to_file("  download timed out (30 min).")
        return False
    except FileNotFoundError:
        log_to_file("  gdown not found. Activate venv-clap first.")
        return False


def download(link, dest):
    """Downloads a file or folder from Google Drive or MediaFire. Retries with multiple fallback methods on failure."""
    dest.mkdir(parents=True, exist_ok=True)
    
    if "mediafire.com" in link:
        return download_mediafire(link, dest)
        
    gdrive_id, is_folder = get_gdrive_id(link)
    
    # Check if a custom cookies file exists
    cookies_path = Path("cookies.txt")
    cookies = None
    if cookies_path.exists():
        log_to_file(f"  cookies.txt found! Using cookies for downloads.")
        cookies = load_cookies_txt(cookies_path)
        
    # Check if rclone Google Drive remote exists
    rclone_remote = get_rclone_drive_remote()
    if rclone_remote:
        log_to_file(f"  rclone Google Drive remote '{rclone_remote}' detected!")
        
    if is_folder:
        # Fallback Method 1: rclone copy (if configured)
        if rclone_remote and gdrive_id:
            log_to_file("Method 1: rclone folder download")
            if run_rclone_cmd(rclone_remote, gdrive_id, dest):
                return True
                
        # Fallback Method 2: Standard gdown folder download
        cmd = ["gdown", "--folder", link]
        if cookies:
            cmd += ["--cookies", str(cookies_path)]
        log_to_file("Method 2: Standard gdown folder download")
        if run_gdown_cmd(cmd, dest):
            return True
            
        # Fallback Method 3: Folder download with no cookies (only if we didn't specify cookies.txt)
        if not cookies:
            cmd = ["gdown", "--folder", "--no-cookies", link]
            log_to_file("Method 3: Folder download with --no-cookies")
            if run_gdown_cmd(cmd, dest):
                return True
                
        # Fallback Method 4: Folder download with --remaining-ok (partially success allowed)
        cmd = ["gdown", "--folder", "--remaining-ok", link]
        if cookies:
            cmd += ["--cookies", str(cookies_path)]
        log_to_file("Method 4: Folder download with --remaining-ok")
        if run_gdown_cmd(cmd, dest):
            return True
            
        # Fallback Method 5: Custom Google Drive folder crawler + requests/curl downloader
        if gdrive_id:
            log_to_file("Method 5: Custom Google Drive recursive folder crawler + requests/curl downloaders")
            if download_folder_crawled(gdrive_id, dest, cookies=cookies):
                return True
            
    else:
        # Fallback Method 1: rclone copy (if configured)
        if rclone_remote and gdrive_id:
            log_to_file("Method 1: rclone file download")
            if run_rclone_cmd(rclone_remote, gdrive_id, dest):
                return True
                
        # Fallback Method 2: Standard gdown file download
        cmd = ["gdown", link]
        if cookies:
            cmd += ["--cookies", str(cookies_path)]
        log_to_file("Method 2: Standard gdown file download")
        if run_gdown_cmd(cmd, dest):
            return True
            
        # Fallback Method 3: File download with no cookies
        if not cookies:
            cmd = ["gdown", "--no-cookies", link]
            log_to_file("Method 3: File download with --no-cookies")
            if run_gdown_cmd(cmd, dest):
                return True
                
        # Fallback Method 4: File download with --continue (resume incomplete transfers)
        cmd = ["gdown", "--continue", link]
        if cookies:
            cmd += ["--cookies", str(cookies_path)]
        log_to_file("Method 4: File download with --continue")
        if run_gdown_cmd(cmd, dest):
            return True
            
        # Fallback Method 5: Python requests direct download with virus warning bypass
        if gdrive_id:
            log_to_file("Method 5: Python requests direct download with virus warning bypass")
            if download_gdrive_file_requests(gdrive_id, dest, cookies=cookies):
                return True
                
        # Fallback Method 6: Native curl download with virus warning bypass
        if gdrive_id:
            log_to_file("Method 6: Native curl download with virus warning bypass")
            if run_curl_download(gdrive_id, dest, cookies_path=cookies_path if cookies else None):
                return True
                
    log_to_file(f"All download fallback methods failed for: {link}")
    return False


# ---- extract ----------------------------------------------------------------
def extract_all(root):
    """Recursively unzip any archives found under root (in place)."""
    for path in list(root.rglob("*")):
        if path.suffix.lower() == ".zip":
            out = path.with_suffix("")
            try:
                with zipfile.ZipFile(path) as z:
                    z.extractall(out)
                path.unlink(missing_ok=True)  # save space
            except Exception as e:
                log_to_file(f"  couldn't unzip {path.name}: {e}")
        elif path.suffix.lower() in (".rar", ".7z"):
            log_to_file(f"  {path.name}: .{path.suffix[1:]} not supported (try `brew install unar` and unpack manually).")


# ---- normalize / sort -------------------------------------------------------
def normalize(root, link, seen):
    """Walk root and sort one-shots, dedup. Returns stats.

    Specific types (label != UNSORTED) go to LABELED_DIR/<label>/ -- training
    gold. Grab-bag matches (label == UNSORTED, e.g. a flat "Percs" or "Misc"
    folder with no specific keyword) go to UNSORTED_DIR instead: they still get
    copied, deduped, and manifested (useful for the map later) but are kept out
    of the trainable classes entirely so they can't muddy or contradict a
    specific label learned elsewhere. See labels.py for the full rationale.
    """
    added = {lbl: 0 for lbl in LABELS}
    added_unsorted = 0
    folder_map = {}   # discovered folder -> label (for the per-kit summary)
    skipped_long = skipped_dupe = skipped_unlabeled = skipped_bad = 0

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue

        label = classify_path(str(path))
        if label is None:
            skipped_unlabeled += 1
            continue

        dur = duration_of(path)
        if dur is None:
            skipped_bad += 1
            continue
        if dur > MAX_ONESHOT_SEC:
            skipped_long += 1
            continue

        h = file_hash(path)
        if h in seen:
            skipped_dupe += 1
            continue

        dest_dir = UNSORTED_DIR if label == UNSORTED else (LABELED_DIR / label)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{h[:12]}{path.suffix.lower()}"
        shutil.copy2(path, dest)

        seen.add(h)
        append_seen(h)
        source_folder = os.path.basename(path.parent)
        append_manifest([h, label, f"{dur:.3f}", path.name, source_folder, link])
        if label == UNSORTED:
            added_unsorted += 1
        else:
            added[label] += 1
        folder_map.setdefault(source_folder, label)

    skips = dict(long=skipped_long, dupe=skipped_dupe,
                 unlabeled=skipped_unlabeled, bad=skipped_bad)
    return added, added_unsorted, folder_map, skips


# ---- display ----------------------------------------------------------------
def print_tally(counts, target):
    sh(f"Progress  (target {target}/category · specific types only)")
    done = 0
    for lbl in sorted(LABELS, key=lambda k: counts[k]):
        n = counts[lbl]
        frac = min(n / target, 1.0) if target else 0
        filled = int(frac * 22)
        bar = "█" * filled + "░" * (22 - filled)
        if n >= target:
            done += 1
            mark, col = "✓", C_GREEN
        elif n == 0:
            mark, col = " ", C_DIM
        else:
            mark, col = " ", C_YEL
        print(f"  {col}{lbl:11s} {n:4d}/{target:<4d} [{bar}] {mark}{C_RST}")
    total = sum(counts.values())
    unsorted_n = unsorted_count()
    print(f"\n  {C_BOLD}{total} labeled one-shots · {done}/{len(LABELS)} categories at target{C_RST}")
    print(f"  {C_DIM}+ {unsorted_n} in unsorted/ (grab-bag folders — Percs, Misc, FX, ... — "
          f"held out of training, used for the map){C_RST}")


# ---- main loop --------------------------------------------------------------
def process(link, target, seen):
    LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed = LINKS_FILE.read_text().split() if LINKS_FILE.exists() else []
    if link in processed:
        log_to_file(f"already processed this link — reprocessing anyway: {link}")

    with tempfile.TemporaryDirectory(dir=RAW_DIR) as tmp:
        tmp = Path(tmp)
        log_to_file(f"\n{C_BOLD}=== Processing Link ==={C_RST}\nURL: {link}")
        log_to_file(f"{C_BOLD}Downloading...{C_RST}")
        if not download(link, tmp):
            log_to_file(f"{C_YEL}Download failed for link:{C_RST} {link}")
            return None
        log_to_file(f"{C_BOLD}Unzipping...{C_RST}")
        extract_all(tmp)
        log_to_file(f"{C_BOLD}Sorting one-shots...{C_RST}")
        added, added_unsorted, folder_map, skips = normalize(tmp, link, seen)

    if folder_map:
        log_to_file(f"{C_BOLD}folders → labels:{C_RST}")
        for folder, lbl in sorted(folder_map.items()):
            log_to_file(f"  {C_DIM}{folder[:38]:38s}{C_RST} → {lbl}")
    got = sum(added.values())
    detail = ", ".join(f"{v} {k}" for k, v in added.items() if v)
    log_to_file(f"  {C_GREEN}+{got} new labeled one-shots{C_RST}" + (f"  ({detail})" if detail else ""))
    if added_unsorted:
        log_to_file(f"  {C_YEL}+{added_unsorted} new unsorted{C_RST}  {C_DIM}(grab-bag folder, no specific type found — held out of training){C_RST}")
    log_to_file(f"  skipped: {skips['dupe']} dupes, {skips['long']} loops, {skips['unlabeled']} unlabeled, {skips['bad']} unreadable")

    with open(LINKS_FILE, "a") as f:
        f.write(link + "\n")
        
    # Write visual tally to log
    log_to_file("\n" + "="*50)
    print_tally_to_log(tally(), target)
    log_to_file("="*50 + "\n")
        
    return got, added_unsorted, skips


def process_local(path_str, target, seen):
    local_path = Path(path_str)
    # We can create a temp directory or just copy/extract
    with tempfile.TemporaryDirectory(dir=RAW_DIR) as tmp:
        tmp = Path(tmp)
        log_to_file(f"\n{C_BOLD}=== Processing Local Path ==={C_RST}\nPath: {local_path}")
        if local_path.is_file():
            log_to_file(f"Copying local file {local_path.name}...")
            shutil.copy2(local_path, tmp / local_path.name)
            log_to_file("Unzipping...")
            extract_all(tmp)
        elif local_path.is_dir():
            log_to_file(f"Copying local folder contents from {local_path.name}...")
            shutil.copytree(local_path, tmp / local_path.name, dirs_exist_ok=True)
            
        log_to_file("Sorting one-shots...")
        added, added_unsorted, folder_map, skips = normalize(tmp, f"local://{local_path.name}", seen)

    if folder_map:
        log_to_file(f"{C_BOLD}folders → labels:{C_RST}")
        for folder, lbl in sorted(folder_map.items()):
            log_to_file(f"  {C_DIM}{folder[:38]:38s}{C_RST} → {lbl}")
    got = sum(added.values())
    detail = ", ".join(f"{v} {k}" for k, v in added.items() if v)
    log_to_file(f"  {C_GREEN}+{got} new labeled one-shots{C_RST}" + (f"  ({detail})" if detail else ""))
    if added_unsorted:
        log_to_file(f"  {C_YEL}+{added_unsorted} new unsorted{C_RST}  {C_DIM}(grab-bag folder, no specific type found — held out of training){C_RST}")
    log_to_file(f"  skipped: {skips['dupe']} dupes, {skips['long']} loops, {skips['unlabeled']} unlabeled, {skips['bad']} unreadable")

    # Write visual tally to log
    log_to_file("\n" + "="*50)
    print_tally_to_log(tally(), target)
    log_to_file("="*50 + "\n")
        
    return got, added_unsorted, skips


def main():
    global current_link
    for d in (DATA_DIR, RAW_DIR, LABELED_DIR, UNSORTED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    seen = load_seen()
    
    # Use a dictionary to share mutable config/state with the background worker
    state = {
        "target": TARGET_PER_CLASS,
        "seen": seen
    }

    def worker():
        global current_link
        while True:
            try:
                link = link_queue.get()
            except Exception:
                break
            
            with current_link_lock:
                current_link = link
                
            try:
                current_target = state["target"]
                is_link = "drive.google.com" in link.lower() or "mediafire.com" in link.lower()
                
                if is_link:
                    print(f"\n{C_BOLD}[Background] Starting to process:{C_RST} {link}")
                    log_to_file(f"Starting background process for link: {link}")
                    res = process(link, current_target, state["seen"])
                else:
                    path_name = os.path.basename(link)
                    print(f"\n{C_BOLD}[Background] Starting to process local file:{C_RST} {path_name}")
                    log_to_file(f"Starting background process for local path: {link}")
                    res = process_local(link, current_target, state["seen"])
                    
                if res is not None:
                    got, added_unsorted, skips = res
                    counts = tally()
                    total = sum(counts.values())
                    done = sum(1 for n in counts.values() if n >= current_target)
                    label = "Finished" if is_link else "Finished local file"
                    name = link if is_link else os.path.basename(link)
                    print(f"\n{C_GREEN}[Background] {label}:{C_RST} {name} -> "
                          f"+{got} labeled, +{added_unsorted} unsorted "
                          f"({total} total, {done}/{len(LABELS)} categories at target)")
                else:
                    label = "Failed" if is_link else "Failed local file"
                    name = link if is_link else os.path.basename(link)
                    print(f"\n{C_YEL}[Background] {label} to process:{C_RST} {name} (see training_data/collect.log for details)")
            except Exception as e:
                print(f"\n{C_YEL}[Background] Error processing {link}: {e}{C_RST} (see training_data/collect.log for details)")
                log_to_file(f"Error processing {link}: {e}\n{traceback.format_exc()}")
            finally:
                with current_link_lock:
                    current_link = None
                link_queue.task_done()
                sys.stdout.write(f"\n{C_BOLD}link ▶ {C_RST}")
                sys.stdout.flush()

    # Start the worker thread
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Open log Terminal
    open_log_terminal()

    print(f"{C_BOLD}PulseMap collector{C_RST} — paste Google Drive or MediaFire links, type 'quit' to stop.")
    print(f"{C_DIM}Labeled sounds go to {LABELED_DIR}/  ·  edit labels.py to tweak sorting.{C_RST}")
    print_tally(tally(), state["target"])

    while True:
        try:
            raw = input(f"\n{C_BOLD}link ▶ {C_RST}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not raw:
            continue
        low = raw.lower()
        if low in ("quit", "q", "exit"):
            if not link_queue.empty() or current_link is not None:
                queued_count = link_queue.qsize()
                active = " (processing 1 link)" if current_link is not None else ""
                print(f"\n{C_YEL}Queue not empty: {queued_count} link(s) remaining{active}.{C_RST}")
                print("Waiting for background tasks to finish. Press Ctrl+C to force exit.")
                try:
                    while not link_queue.empty() or current_link is not None:
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    print("\nForced exit. bye.")
                    break
            print("bye.")
            break
        if low == "status":
            queued_count = link_queue.qsize()
            with current_link_lock:
                curr = current_link
            
            sh("Background Queue Status")
            if curr:
                print(f"  {C_GREEN}Active task:{C_RST} {curr}")
            else:
                print("  Active task: None")
            print(f"  {C_BOLD}Queued links:{C_RST} {queued_count}")
            
            print_tally(tally(), state["target"])
            continue
        if low == "open":
            print(f"  labeled:  {LABELED_DIR.resolve()}")
            print(f"  unsorted: {UNSORTED_DIR.resolve()}")
            continue
        if low.startswith("target"):
            parts = raw.split()
            if len(parts) == 2 and parts[1].isdigit():
                new_target = int(parts[1])
                state["target"] = new_target
                print(f"  target set to {new_target}/category.")
            else:
                print("  usage: target <number>")
            continue
            
        # Clean local paths / check link
        if "http://" in raw.lower() or "https://" in raw.lower():
            # Extract only the URL part
            match = re.search(r'https?://[^\s]+', raw)
            cleaned = match.group(0).strip() if match else raw.strip()
        else:
            cleaned = clean_local_path(raw)
            
        cleaned_low = cleaned.lower()
        if "drive.google.com" not in cleaned_low and "mediafire.com" not in cleaned_low:
            local_path = Path(cleaned)
            if not local_path.exists():
                print(f"{C_YEL}  that doesn't look like a Google Drive/MediaFire link or a valid local path.{C_RST}")
                continue

        link_queue.put(cleaned)
        q_size = link_queue.qsize()
        with current_link_lock:
            curr = current_link
        active_status = " (processing starts shortly)" if not curr else f" (currently processing {curr[:40]}...)"
        print(f"  {C_GREEN}Added to queue.{C_RST} Position in queue: {q_size}{active_status}")


if __name__ == "__main__":
    main()
