"""EF101-P16: Local LLM Hosting and Evaluation System

A Tkinter frontend for local Ollama models with persistent conversation history,
streaming responses, a fixed-prompt benchmark, and CPU/RAM monitoring.
Core inference uses the local Ollama API; no cloud LLM API is used.

Requirements: Python 3.10+, Ollama, and psutil (optional but recommended).
Install psutil with: python -m pip install psutil
"""
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, scrolledtext, ttk

try:
    import psutil
except ImportError:
    psutil = None

OLLAMA_URL = "http://localhost:11434"
CHAT_URL = OLLAMA_URL + "/api/chat"
TAGS_URL = OLLAMA_URL + "/api/tags"
BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "history.json"
BENCHMARK_FILE = BASE_DIR / "benchmark_results.csv"
LOAD_FILE = BASE_DIR / "model_load_tests.csv"
CONVO_CSV = BASE_DIR / "conversation_log.csv"
MODEL_INFO_FILE = BASE_DIR / "model_info.csv"
SUMMARY_FILE = BASE_DIR / "benchmark_summary.csv"
SUMMARY_PROMPT_FILE = BASE_DIR / "benchmark_summary_by_prompt.csv"
SHOW_URL = OLLAMA_URL + "/api/show"
PS_URL = OLLAMA_URL + "/api/ps"
REQUEST_TIMEOUT = 600
MAX_HISTORY = 200
CONTEXT_MESSAGES = 20
GENERATION_OPTIONS = {"temperature": 0.2, "num_predict": 500}
DEFAULT_MODELS = ["gemma3:4b", "llama3.2:3b"]
FONT = "Segoe UI"
CODE_FONT = "Consolas"

CHAT_LENGTH_INSTRUCTION = (
    "Keep replies concise and direct. Usually use 2–4 short sentences or a few bullets. "
    "Avoid long introductions, repetition, and unnecessary background. "
    "For calculations, show only essential steps. For coding requests, provide the complete requested code but keep the explanation brief. "
    "If the user explicitly asks for detail, then expand."
)
STYLES = {
    "None (default)": "",
    "Short answer": "Answer in at most three short sentences unless code or essential steps are required.",
    "Explain simply": "Explain for a beginner and include one everyday example.",
    "Step by step": "Give a clear numbered solution with the necessary steps.",
}
# The same ten fixed prompts are available both as clickable chat shortcuts and
# in the benchmark. Clicking a prompt loads it into the chat box; it does not
# silently submit it, so the user can review/edit it before sending.
QUICK_PROMPTS = [
    (pid, category, prompt) for pid, category, prompt in [
        ("P01", "Explanation", "Explain the difference between RAM and storage in simple terms. Give one example of each."),
        ("P02", "Electrical reasoning", "Using Kirchhoff's Current Law, find Ix if 5 A enters a node and currents of 2 A and Ix leave it. Show the equation and answer."),
        ("P03", "Coding", "Write a Python function named is_prime(n) that returns True if n is prime and False otherwise. Include code that handles numbers less than 2."),
        ("P04", "Numerical reasoning", "A 4-ohm resistor is connected across an ideal 12 V source. Calculate the current using Ohm's law. Show the formula, substitution, and unit."),
        ("P05", "Summarization", "Summarize this paragraph in exactly two sentences: 'Electrical energy can be generated using different sources, including fossil fuels, wind, solar, and water. Power stations convert energy into electrical energy, which is transmitted through power lines. Transformers help adjust voltage for efficient transmission and safe distribution to consumers.'"),
        ("P06", "Rewriting", "Rewrite this sentence in clearer, simpler English without changing its meaning: 'The implementation of energy-efficient practices can significantly reduce the consumption of electrical power in residential and commercial environments.'"),
        ("P07", "Structured output", "Return ONLY valid JSON with keys 'voltage_V', 'resistance_ohms', and 'current_A'. Use voltage 12 V and resistance 4 ohms. Calculate current using Ohm's law."),
        ("P08", "Number systems", "Convert binary 1010 to decimal. Show the place-value calculation."),
        ("P09", "Coding/debugging", "Fix the syntax error in this Python code and briefly explain the fix:\nfor i in range(3)\n    print(i)"),
        ("P10", "Technical reasoning", "For an ideal resistor connected to a fixed-voltage source, explain what happens to current if resistance increases. Use Ohm's law."),
    ]
]
PROMPTS = [
    ("P01", "Explanation", "Explain the difference between RAM and storage in simple terms. Give one example of each."),
    ("P02", "Electrical reasoning", "Using Kirchhoff's Current Law, find Ix if 5 A enters a node and currents of 2 A and Ix leave it. Show the equation and answer."),
    ("P03", "Coding", "Write a Python function named is_prime(n) that returns True if n is prime and False otherwise. Include code that handles numbers less than 2."),
    ("P04", "Numerical reasoning", "A 4-ohm resistor is connected across an ideal 12 V source. Calculate the current using Ohm's law. Show the formula, substitution, and unit."),
    ("P05", "Summarization", "Summarize this paragraph in exactly two sentences: 'Electrical energy can be generated using different sources, including fossil fuels, wind, solar, and water. Power stations convert energy into electrical energy, which is transmitted through power lines. Transformers help adjust voltage for efficient transmission and safe distribution to consumers.'"),
    ("P06", "Rewriting", "Rewrite this sentence in clearer, simpler English without changing its meaning: 'The implementation of energy-efficient practices can significantly reduce the consumption of electrical power in residential and commercial environments.'"),
    ("P07", "Structured output", "Return ONLY valid JSON with keys 'voltage_V', 'resistance_ohms', and 'current_A'. Use voltage 12 V and resistance 4 ohms. Calculate current using Ohm's law."),
    ("P08", "Number systems", "Convert binary 1010 to decimal. Show the place-value calculation."),
    ("P09", "Coding/debugging", "Fix the syntax error in this Python code and briefly explain the fix:\nfor i in range(3)\n    print(i)"),
    ("P10", "Technical reasoning", "For an ideal resistor connected to a fixed-voltage source, explain what happens to current if resistance increases. Use Ohm's law."),
]
BENCH_COLUMNS = [
    "timestamp", "trial_id", "trial_number", "model", "model_file_size", "parameter_size", "quantization",
    "model_family", "model_format", "context_length", "prompt_id", "prompt_category", "prompt",
    "generation_temperature", "generation_num_predict", "response_time_seconds", "load_duration_seconds",
    "ollama_total_duration_seconds", "prompt_eval_count", "eval_count", "eval_duration_seconds",
    "tokens_per_second", "done_reason", "resource_samples", "system_cpu_avg_percent", "system_cpu_max_percent",
    "ollama_cpu_avg_percent_of_total_capacity", "ollama_cpu_max_percent_of_total_capacity",
    "ollama_ram_avg_mb", "ollama_ram_max_mb", "detected_ollama_processes_max",
    "gpu_util_avg_percent", "gpu_util_max_percent", "gpu_mem_max_mb",
    "loaded_model_mb", "model_vram_mb", "gpu_offload_percent",
    "status", "answer", "error", "correctness_score", "relevance_score", "completeness_score", "evaluation_notes",
]
LOAD_COLUMNS = [
    "timestamp", "model", "model_file_size", "parameter_size", "quantization", "load_test_type",
    "response_time_seconds", "load_duration_seconds", "ollama_total_duration_seconds",
    "prompt_eval_count", "eval_count", "done_reason", "status", "error",
]
CONVO_COLUMNS = [
    "timestamp", "model", "parameter_size", "quantization", "style", "prompt", "response",
    "response_time_sec", "first_token_sec", "load_time_sec", "eval_count", "tokens_per_sec", "stopped",
]
MODEL_INFO_COLUMNS = [
    "model", "source", "parameter_size", "quantization", "family", "format",
    "context_length", "file_size", "license", "recorded_at",
]

PALETTES = {
    "Indigo": {"bg":"#0f172a", "panel":"#1e293b", "text":"#e2e8f0", "muted":"#94a3b8", "accent":"#6366f1", "user":"#4338ca", "ai":"#273449", "border":"#334155", "error":"#f87171"},
    "Light": {"bg":"#f1f5f9", "panel":"#ffffff", "text":"#0f172a", "muted":"#64748b", "accent":"#4f46e5", "user":"#4f46e5", "ai":"#e2e8f0", "border":"#cbd5e1", "error":"#b91c1c"},
    "Emerald": {"bg":"#06201a", "panel":"#0b2e25", "text":"#d1fae5", "muted":"#7dd3c0", "accent":"#10b981", "user":"#0f766e", "ai":"#0e3a2f", "border":"#115e59", "error":"#f87171"},
}


def read_history():
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError):
        pass
    try:
        if HISTORY_FILE.exists():
            backup = HISTORY_FILE.with_suffix(".json.bak")
            if backup.exists():
                backup = HISTORY_FILE.with_name(f"history_{int(time.time())}.json.bak")
            HISTORY_FILE.replace(backup)
    except OSError:
        pass
    return []


def save_history(entries):
    tmp = HISTORY_FILE.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(entries[-MAX_HISTORY:], ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(HISTORY_FILE)
        return True
    except OSError:
        return False


def ensure_csv(path, columns):
    """Create the CSV. If an old file has a different layout, archive it instead of failing.
    Returns the archive path when one was made, else None."""
    archived = None
    if path.exists() and path.stat().st_size:
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as f:
                old = next(csv.reader(f), [])
        except OSError:
            old = columns
        if old != columns:
            archived = path.with_name(f"{path.stem}_old_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
            path.replace(archived)
    if not path.exists() or not path.stat().st_size:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
    return archived


def append_csv(path, columns, row):
    """Append one row. If the file is locked (usually open in Excel), keep the row in
    <name>_unsaved.csv so no data is lost. Returns the path that was actually written."""
    for _ in range(3):
        try:
            with path.open("a", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=columns, extrasaction="ignore").writerow(row)
            return path
        except PermissionError:
            time.sleep(0.5)
    spare = path.with_name(f"{path.stem}_unsaved{path.suffix}")
    is_new = not spare.exists() or not spare.stat().st_size
    with spare.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)
    return spare


def log_conversation_csv(entry, info):
    ensure_csv(CONVO_CSV, CONVO_COLUMNS)
    row = dict(entry)
    row["parameter_size"] = info.get("parameter_size", "")
    row["quantization"] = info.get("quantization", "")
    return append_csv(CONVO_CSV, CONVO_COLUMNS, row)


def installed_models():
    req = urllib.request.Request(TAGS_URL)
    with urllib.request.urlopen(req, timeout=5) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("models", [])


def model_size(model_name):
    try:
        for item in installed_models():
            if item.get("name") == model_name:
                size = item.get("size", 0)
                return f"{size / 1e9:.2f} GB" if size else ""
    except Exception:
        pass
    if shutil.which("ollama"):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            for line in result.stdout.splitlines()[1:]:
                fields = line.split()
                if len(fields) >= 3 and fields[0] == model_name:
                    return " ".join(fields[2:4])
        except Exception:
            pass
    return ""


def ns_seconds(value):
    try:
        return round(float(value) / 1e9, 4)
    except (TypeError, ValueError):
        return ""


def api_stats(data):
    eval_s = ns_seconds(data.get("eval_duration"))
    count = data.get("eval_count")
    tps = round(count / eval_s, 2) if (eval_s and count) else ""
    return {"load_duration_seconds": ns_seconds(data.get("load_duration")),
            "ollama_total_duration_seconds": ns_seconds(data.get("total_duration")),
            "prompt_eval_count": data.get("prompt_eval_count", ""),
            "eval_count": data.get("eval_count", ""),
            "eval_duration_seconds": eval_s,
            "tokens_per_second": tps,
            "done_reason": data.get("done_reason", "")}


_DETAILS_CACHE = {}


def model_details(model, refresh=False):
    """Name, source, parameter count, quantization, file size, context length (from Ollama /api/show)."""
    if model in _DETAILS_CACHE and not refresh:
        return _DETAILS_CACHE[model]
    source = model if model.startswith("hf.co") else "Ollama library: ollama.com/library/" + model.split(":")[0]
    info = {"model": model, "source": source, "parameter_size": "", "quantization": "", "family": "",
            "format": "", "context_length": "", "file_size": model_size(model), "license": "",
            "recorded_at": datetime.now().isoformat(timespec="seconds")}
    try:
        req = urllib.request.Request(SHOW_URL, data=json.dumps({"model": model, "name": model}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return info  # not cached, so it is retried next time
    d = data.get("details") or {}
    info.update(parameter_size=d.get("parameter_size", ""), quantization=d.get("quantization_level", ""),
                family=d.get("family", ""), format=d.get("format", ""))
    for key, value in (data.get("model_info") or {}).items():
        if key.endswith(".context_length"):
            info["context_length"] = value
            break
    lines = (data.get("license") or "").strip().splitlines()
    info["license"] = lines[0].strip() if lines else ""
    _DETAILS_CACHE[model] = info
    return info


def save_model_info(models):
    """Write one row per model to model_info.csv (a snapshot, overwritten each time)."""
    with MODEL_INFO_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=MODEL_INFO_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for name in models:
            w.writerow(model_details(name))
    return MODEL_INFO_FILE


def loaded_info(model):
    """Ask Ollama how much of the loaded model sits in RAM vs GPU memory (/api/ps)."""
    out = {"loaded_model_mb": "", "model_vram_mb": "", "gpu_offload_percent": ""}
    try:
        with urllib.request.urlopen(PS_URL, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        for m in data.get("models", []):
            if model in (m.get("name"), m.get("model")):
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                out.update(loaded_model_mb=round(size / 1048576, 1), model_vram_mb=round(vram / 1048576, 1),
                           gpu_offload_percent=round(100 * vram / size, 1) if size else "")
                break
    except Exception:
        pass
    return out


NVIDIA_SMI = shutil.which("nvidia-smi")


def gpu_sample():
    """(utilization %, memory MB) from nvidia-smi, or None when no NVIDIA GPU is available."""
    if not NVIDIA_SMI:
        return None
    try:
        out = subprocess.run([NVIDIA_SMI, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip().splitlines()
        util, mem = [float(x) for x in out[0].split(",")]
        return util, mem
    except Exception:
        return None


def _floats(rows, key):
    out = []
    for r in rows:
        try:
            if r.get(key) not in (None, ""):
                out.append(float(r[key]))
        except ValueError:
            pass
    return out


def _mean(v): return round(sum(v) / len(v), 3) if v else ""
def _min(v): return round(min(v), 3) if v else ""
def _max(v): return round(max(v), 3) if v else ""


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary():
    """Average / range per model and per prompt from the scored benchmark CSV."""
    if not BENCHMARK_FILE.exists():
        raise RuntimeError("No benchmark CSV exists yet. Run a benchmark first.")
    with BENCHMARK_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("The benchmark CSV has no data rows yet.")
    cold = {}
    if LOAD_FILE.exists():
        with LOAD_FILE.open("r", newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "success":
                    cold.setdefault(r.get("model", ""), []).extend(_floats([r], "load_duration_seconds"))

    def block(subset):
        ok = [r for r in subset if r.get("status") == "success"]
        rt = _floats(ok, "response_time_seconds")
        return ok, {
            "trials_recorded": len(subset), "successful": len(ok), "errors": len(subset) - len(ok),
            "truncated_at_token_limit": sum(1 for r in ok if r.get("done_reason") == "length"),
            "response_time_mean_s": _mean(rt), "response_time_min_s": _min(rt), "response_time_max_s": _max(rt),
            "tokens_per_second_mean": _mean(_floats(ok, "tokens_per_second")),
            "ram_mean_mb": _mean(_floats(ok, "ollama_ram_avg_mb")),
            "ram_peak_mb": _max(_floats(ok, "ollama_ram_max_mb")),
            "ollama_cpu_mean_percent": _mean(_floats(ok, "ollama_cpu_avg_percent_of_total_capacity")),
            "ollama_cpu_peak_percent": _max(_floats(ok, "ollama_cpu_max_percent_of_total_capacity")),
            "system_cpu_mean_percent": _mean(_floats(ok, "system_cpu_avg_percent")),
            "gpu_util_mean_percent": _mean(_floats(ok, "gpu_util_avg_percent")),
            "gpu_offload_mean_percent": _mean(_floats(ok, "gpu_offload_percent")),
            "correctness_mean": _mean(_floats(ok, "correctness_score")),
            "relevance_mean": _mean(_floats(ok, "relevance_score")),
            "completeness_mean": _mean(_floats(ok, "completeness_score")),
        }

    per_model = []
    for m in sorted({r.get("model", "") for r in rows}):
        subset = [r for r in rows if r.get("model") == m]
        _, stats = block(subset)
        head = {"model": m, "parameter_size": subset[0].get("parameter_size", ""),
                "quantization": subset[0].get("quantization", ""), "model_file_size": subset[0].get("model_file_size", ""),
                "cold_load_time_mean_s": _mean(cold.get(m, []))}
        per_model.append({**head, **stats})

    per_prompt = []
    for key in sorted({(r.get("prompt_id", ""), r.get("model", "")) for r in rows}):
        subset = [r for r in rows if (r.get("prompt_id", ""), r.get("model", "")) == key]
        _, stats = block(subset)
        per_prompt.append({"prompt_id": key[0], "prompt_category": subset[0].get("prompt_category", ""),
                           "model": key[1], **stats})

    _write_rows(SUMMARY_FILE, per_model)
    _write_rows(SUMMARY_PROMPT_FILE, per_prompt)
    return f"Summary saved:\n{SUMMARY_FILE}\n{SUMMARY_PROMPT_FILE}"


class ResourceMonitor:
    def __init__(self, interval=0.75):
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = []
        self.procs = {}
        self.thread = None
        self.logical = (psutil.cpu_count(logical=True) or 1) if psutil else 1

    def sample(self):
        if not psutil:
            return
        try:
            sys_cpu = psutil.cpu_percent(interval=None)
            for p in psutil.process_iter(["pid", "name", "exe"]):
                try:
                    info = p.info
                    if "ollama" in ((info.get("name") or "") + " " + (info.get("exe") or "")).lower() or "llama-server" in ((info.get("name") or "") + " " + (info.get("exe") or "")).lower():
                        self.procs[p.pid] = p
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            cpu = ram = count = 0
            g = gpu_sample()
            for pid, p in list(self.procs.items()):
                try:
                    cpu += p.cpu_percent(interval=None)
                    ram += p.memory_info().rss
                    count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.procs.pop(pid, None)
            self.samples.append({"time": datetime.now().isoformat(timespec="seconds"), "system_cpu_percent": round(sys_cpu, 2), "ollama_cpu_percent_of_total_capacity": round(cpu / self.logical, 2), "ollama_process_ram_mb": round(ram / (1024 * 1024), 2), "detected_process_count": count, "gpu_util_percent": g[0] if g else None, "gpu_mem_mb": g[1] if g else None})
        except Exception:
            pass

    def run(self):
        if psutil:
            psutil.cpu_percent(interval=None)
        while not self.stop_event.wait(self.interval):
            self.sample()

    def start(self):
        if psutil:
            self.sample()
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if psutil:
            self.sample()

    def summary(self):
        def avg(key):
            values = [s[key] for s in self.samples]
            return round(sum(values) / len(values), 2) if values else ""
        def maximum(key):
            values = [s[key] for s in self.samples]
            return round(max(values), 2) if values else ""
        gu = [s["gpu_util_percent"] for s in self.samples if s.get("gpu_util_percent") is not None]
        gm = [s["gpu_mem_mb"] for s in self.samples if s.get("gpu_mem_mb") is not None]
        return {"gpu_util_avg_percent": round(sum(gu) / len(gu), 2) if gu else "",
                "gpu_util_max_percent": max(gu) if gu else "", "gpu_mem_max_mb": max(gm) if gm else "",
                "resource_samples": json.dumps(self.samples),
                "system_cpu_avg_percent": avg("system_cpu_percent"), "system_cpu_max_percent": maximum("system_cpu_percent"),
                "ollama_cpu_avg_percent_of_total_capacity": avg("ollama_cpu_percent_of_total_capacity"), "ollama_cpu_max_percent_of_total_capacity": maximum("ollama_cpu_percent_of_total_capacity"),
                "ollama_ram_avg_mb": avg("ollama_process_ram_mb"), "ollama_ram_max_mb": maximum("ollama_process_ram_mb"),
                "detected_ollama_processes_max": max([s["detected_process_count"] for s in self.samples], default="")}


def chat_request(model, messages, stop_event=None, on_chunk=None, stream=True):
    payload = {"model": model, "messages": messages, "stream": stream, "options": GENERATION_OPTIONS}
    request = urllib.request.Request(CHAT_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    pieces = []
    final = {}
    first_token = None
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        if not stream:
            data = json.loads(response.read().decode("utf-8"))
            answer = (data.get("message") or {}).get("content", "")
            if not answer.strip():
                raise RuntimeError("Ollama returned an empty response.")
            return answer, data, time.perf_counter() - started, None
        for raw in response:
            if stop_event and stop_event.is_set():
                break
            if not raw.strip():
                continue
            try:
                packet = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if packet.get("error"):
                raise RuntimeError(packet["error"])
            piece = (packet.get("message") or {}).get("content", "")
            if piece:
                if first_token is None:
                    first_token = time.perf_counter() - started
                pieces.append(piece)
                if on_chunk:
                    on_chunk(piece)
            if packet.get("done"):
                final = packet
                break
    return "".join(pieces), final, time.perf_counter() - started, first_token


def friendly_error(exc, model):
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return f"Model '{model}' was not found. Install it in a terminal with: ollama pull {model}"
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", "")
        except Exception:
            detail = ""
        return f"Ollama returned HTTP {exc.code}. {detail}".strip()
    if isinstance(exc, (urllib.error.URLError, ConnectionError)):
        return "Cannot connect to Ollama. Start the Ollama application and try again."
    if isinstance(exc, TimeoutError):
        return "The request timed out. Try again or use a smaller model."
    return f"Request failed: {type(exc).__name__}: {exc}"


class App:
    def __init__(self, root):
        self.root = root
        root.title("EF101-P16 | Local LLM Chat and Evaluation")
        root.geometry("1080x800")
        root.minsize(850, 620)
        self.theme_name = "Indigo"
        self.P = PALETTES[self.theme_name]
        self.font_size = 11
        self.messages = []
        self.history = read_history()
        self.q = queue.Queue()
        self.busy = False
        self.benchmarking = False
        self.bench_label = ""
        self.stop_event = threading.Event()
        self.ai_started = False
        self.last_reply = ""
        self.times = []
        self.placeholder_on = False
        self.widgets = []
        self.live_processes = {}
        self.style = ttk.Style()
        try: self.style.theme_use("clam")
        except tk.TclError: pass
        self.build_ui()
        self.apply_theme()
        self.show_placeholder()
        self.welcome()
        root.bind("<Control-n>", lambda _e: self.new_conversation())
        root.bind("<Escape>", lambda _e: self.stop())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_models()
        self.refresh_csv_status()
        self.poll_queue()
        self.check_runtime_periodically()
        self.update_live_resources()
        self.animate()

    def reg(self, widget, **opts):
        self.widgets.append((widget, opts)); return widget

    def button(self, parent, text, command, color="normal"):
        b = tk.Button(parent, text=text, command=command, relief="flat", bd=0, padx=10, pady=6, cursor="hand2", font=(FONT, 9))
        b._color = color
        self.reg(b, bg={"normal": "border"}.get(color, color), fg="text", activebackground="muted", activeforeground="bg")
        return b

    def build_ui(self):
        r = self.root
        r.configure(bg=self.P["bg"])
        r.columnconfigure(0, weight=1); r.rowconfigure(2, weight=1)
        header = self.reg(tk.Frame(r, padx=16, pady=10), bg="panel")
        header.grid(row=0, column=0, sticky="ew"); header.columnconfigure(0, weight=1)
        self.reg(tk.Label(header, text="✦  Local LLM Chat", font=(FONT, 18, "bold")), bg="panel", fg="text").grid(row=0, column=0, sticky="w")
        self.runtime_label = self.reg(tk.Label(header, text="Ollama: checking…", font=(FONT, 9, "bold")), bg="panel", fg="muted")
        self.runtime_label.grid(row=0, column=1, padx=8)
        self.theme_var = tk.StringVar(value=self.theme_name)
        ttk.Combobox(header, textvariable=self.theme_var, values=list(PALETTES), state="readonly", width=10).grid(row=0, column=2, padx=4)
        self.theme_var.trace_add("write", lambda *_: self.set_theme())
        self.zoom_label = tk.Label(header, text="Text size", font=(FONT, 9)); self.reg(self.zoom_label, bg="panel", fg="text").grid(row=0, column=3, padx=(10, 3))
        self.button(header, "A−", lambda: self.zoom(-1)).grid(row=0, column=4, padx=2)
        self.button(header, "A+", lambda: self.zoom(1)).grid(row=0, column=5, padx=2)

        bar = self.reg(tk.Frame(r, padx=14, pady=10), bg="panel"); bar.grid(row=1, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)

        # Row 0: model / style selectors on the left, saved-results counter on the right.
        sel = self.reg(tk.Frame(bar), bg="panel"); sel.grid(row=0, column=0, sticky="ew")
        self.reg(tk.Label(sel, text="Model", font=(FONT, 9, "bold")), bg="panel", fg="text").pack(side="left", padx=(0, 6))
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(sel, textvariable=self.model_var, state="readonly", width=22)
        self.model_box.pack(side="left"); self.model_box.bind("<<ComboboxSelected>>", self.model_changed)
        self.reg(tk.Label(sel, text="Style", font=(FONT, 9, "bold")), bg="panel", fg="text").pack(side="left", padx=(18, 6))
        self.style_var = tk.StringVar(value="Short answer")
        self.style_box = ttk.Combobox(sel, textvariable=self.style_var, values=list(STYLES), state="readonly", width=17)
        self.style_box.pack(side="left")
        self.button(sel, "Refresh models", self.refresh_models).pack(side="left", padx=(18, 0))
        self.csv_status = self.reg(tk.Label(sel, text="Benchmark results: no scored rows yet", font=(FONT, 9)), bg="panel", fg="muted")
        self.csv_status.pack(side="right")

        # Rows 1-2: current model and its details.
        self.current_label = self.reg(tk.Label(bar, text="Current model: —", font=(FONT, 10, "bold"), anchor="w"), bg="panel", fg="text")
        self.current_label.grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.info_label = self.reg(tk.Label(bar, text="Model details: —", font=(FONT, 9), anchor="w", justify="left", wraplength=900), bg="panel", fg="muted")
        self.info_label.grid(row=2, column=0, sticky="w", pady=(2, 0))
        bar.bind("<Configure>", lambda e: self.info_label.configure(wraplength=max(300, e.width - 40)))

        # Rows 3-4: two button rows with equal-width columns so everything lines up.
        def button_row(row, caption, items):
            frame = self.reg(tk.Frame(bar), bg="panel"); frame.grid(row=row, column=0, sticky="ew", pady=(10, 0))
            self.reg(tk.Label(frame, text=caption, font=(FONT, 9, "bold"), width=11, anchor="w"), bg="panel", fg="muted").grid(row=0, column=0, sticky="w")
            for i, (text, command, color) in enumerate(items):
                self.button(frame, text, command, color).grid(row=0, column=i + 1, sticky="ew", padx=3)
                frame.columnconfigure(i + 1, weight=1, uniform="btn")
        button_row(3, "CHAT", [
            ("New conversation", self.new_conversation, "normal"),
            ("Clear history", self.clear_history, "error"),
            ("View history", self.view_history, "normal"),
            ("Export history CSV", self.export_history, "normal"),
            ("Copy reply", self.copy_reply, "normal"),
        ])
        button_row(4, "BENCHMARK", [
            ("Run benchmark", self.start_benchmark, "accent"),
            ("Summarize results", self.summarize_results, "normal"),
            ("View benchmark CSV", self.view_benchmark_csv, "normal"),
            ("Open results folder", self.open_results_folder, "normal"),
            ("Save model info", self.export_model_info, "normal"),
        ])

        chat_frame = tk.Frame(r)
        chat_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=(8,4))
        chat_frame.columnconfigure(0, weight=1); chat_frame.rowconfigure(0, weight=1)
        self.chat = tk.Text(chat_frame, wrap="word", state="disabled", font=(FONT, self.font_size), relief="flat", padx=14, pady=12, highlightthickness=1)
        self.chat.grid(row=0, column=0, sticky="nsew")
        self.chat_scroll = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat.yview, style="Chat.Vertical.TScrollbar")
        self.chat_scroll.grid(row=0, column=1, sticky="ns")
        self.chat.configure(yscrollcommand=self.chat_scroll.set)
        self.stats = self.reg(tk.Label(r, text="Response: —   First token: —   Speed: —   Model load: —", anchor="w", font=(FONT, 9)), bg="bg", fg="muted")
        self.stats.grid(row=3, column=0, sticky="ew", padx=14, pady=(3, 1))
        self.live_resources = self.reg(tk.Label(r, text="Live resources: collecting CPU/RAM data…", anchor="w", font=(FONT, 9, "bold")), bg="bg", fg="muted")
        self.live_resources.grid(row=4, column=0, sticky="ew", padx=14, pady=(1, 4))
        quick = self.reg(tk.Frame(r, padx=12, pady=3), bg="bg"); quick.grid(row=5, column=0, sticky="ew")
        self.reg(tk.Label(quick, text="10 fixed prompts — click one to load it into the chat box:", font=(FONT, 9, "bold")), bg="bg", fg="muted").grid(row=0, column=0, columnspan=5, sticky="w", pady=(0,3))
        for index, (pid, category, prompt_text) in enumerate(QUICK_PROMPTS):
            row_index = 1 + index // 5
            col_index = index % 5
            self.button(quick, f"{pid} · {category}", lambda p=prompt_text: self.quick_prompt(p)).grid(row=row_index, column=col_index, sticky="ew", padx=2, pady=2)
        for column in range(5): quick.columnconfigure(column, weight=1, uniform="q")
        inp = self.reg(tk.Frame(r, padx=12, pady=8), bg="bg"); inp.grid(row=6, column=0, sticky="ew"); inp.columnconfigure(0, weight=1)
        self.prompt = self.reg(tk.Text(inp, height=3, wrap="word", font=(FONT, self.font_size), relief="flat", padx=8, pady=6), bg="panel", fg="text")
        self.prompt.grid(row=0, column=0, sticky="ew", padx=(0,8)); self.prompt.bind("<Return>", self.on_enter); self.prompt.bind("<FocusIn>", self.hide_placeholder); self.prompt.bind("<FocusOut>", lambda _e: self.show_placeholder())
        actions = tk.Frame(inp, bg=self.P["bg"]); actions.grid(row=0, column=1, sticky="ns")
        self.send_btn = self.button(actions, "Send ➤", self.send, "accent"); self.send_btn.pack(fill="both", expand=True)
        self.stop_btn = self.button(actions, "■ Stop", self.stop, "error"); self.stop_btn.pack(fill="both", expand=True); self.stop_btn.pack_forget()

    def apply_theme(self):
        p = self.P; self.root.configure(bg=p["bg"])
        for widget, opts in self.widgets:
            try:
                config = {k: (p[v] if v in p else v) for k, v in opts.items()}
                widget.configure(**config)
            except tk.TclError: pass
        self.chat.configure(bg=p["bg"], fg=p["text"], insertbackground=p["text"], highlightbackground=p["border"])
        self.style.configure("Chat.Vertical.TScrollbar", background=p["border"], troughcolor=p["panel"], bordercolor=p["panel"], arrowcolor=p["text"], lightcolor=p["border"], darkcolor=p["border"])
        self.style.map("Chat.Vertical.TScrollbar", background=[("active", p["accent"]), ("pressed", p["accent"])])
        self.chat.tag_config("user", background=p["user"], foreground="#ffffff", lmargin1=100, lmargin2=100, spacing1=7, spacing3=5)
        self.chat.tag_config("assistant", background=p["ai"], foreground=p["text"], lmargin1=5, lmargin2=5, rmargin=80, spacing1=7, spacing3=5)
        self.chat.tag_config("label", foreground=p["accent"], font=(FONT, self.font_size-1, "bold"))
        self.chat.tag_config("info", foreground=p["muted"], font=(FONT, self.font_size-1, "italic"))
        self.chat.tag_config("error", foreground=p["error"])
        self.chat.tag_config("bold", font=(FONT, self.font_size, "bold"))
        self.chat.tag_config("code", font=(CODE_FONT, self.font_size-1), background=p["panel"])
        self.prompt.configure(bg=p["panel"], fg=p["muted"] if self.placeholder_on else p["text"], insertbackground=p["text"])
        self.stats.configure(bg=p["bg"], fg=p["muted"])

    def set_theme(self):
        name = self.theme_var.get()
        if name in PALETTES:
            self.theme_name = name; self.P = PALETTES[name]; self.apply_theme()

    def zoom(self, delta):
        self.font_size = max(9, min(18, self.font_size + delta)); self.chat.configure(font=(FONT, self.font_size)); self.prompt.configure(font=(FONT, self.font_size)); self.apply_theme()

    def put(self, text, tag=None):
        self.chat.configure(state="normal"); self.chat.insert("end", text, (tag,) if tag else ()); self.chat.see("end"); self.chat.configure(state="disabled")

    def clear_screen(self):
        self.chat.configure(state="normal"); self.chat.delete("1.0", "end"); self.chat.configure(state="disabled")

    def welcome(self):
        self.put("Welcome! Your local AI assistant is ready.\n", "label")
        self.put("Choose a model, enter a prompt, and press Send. Inference uses the local Ollama runtime.\n", "info")
        if self.history: self.put(f"Loaded {len(self.history)} saved history entries. Use View history to inspect them.\n", "info")

    def show_placeholder(self):
        if self.prompt.get("1.0", "end").strip() == "" and not self.placeholder_on:
            self.placeholder_on = True; self.prompt.insert("1.0", "Type your question here… (Enter to send, Shift+Enter for a new line)"); self.prompt.configure(fg=self.P["muted"])

    def hide_placeholder(self, _event=None):
        if self.placeholder_on:
            self.placeholder_on = False; self.prompt.delete("1.0", "end"); self.prompt.configure(fg=self.P["text"])

    def get_prompt(self): return "" if self.placeholder_on else self.prompt.get("1.0", "end").strip()

    def quick_prompt(self, text):
        if self.busy: return
        self.hide_placeholder(); self.prompt.delete("1.0", "end"); self.prompt.insert("1.0", text); self.prompt.focus_set()

    def refresh_models(self):
        self.runtime_label.configure(text="Ollama: checking…")
        threading.Thread(target=self._fetch_models_worker, daemon=True).start()

    def _fetch_models_worker(self):
        try:
            items = installed_models()
            self.q.put(("models", items, None))
            save_model_info([x.get("name") for x in items if x.get("name")])
        except Exception as exc:
            self.q.put(("models", [], exc))

    def apply_models(self, items, error):
        if error:
            self.runtime_label.configure(text="Ollama: not running")
            names = DEFAULT_MODELS
            self.put("Cannot reach Ollama. Start Ollama, then click Refresh models.\n", "error")
        else:
            self.runtime_label.configure(text="Ollama: running")
            names = [x.get("name") for x in items if x.get("name")]
            if not names:
                self.put("Ollama is running, but no models are installed. Use `ollama pull gemma3:4b` and `ollama pull llama3.2:3b` in a terminal.\n", "error")
                names = DEFAULT_MODELS
        self.model_box["values"] = names
        if self.model_var.get() not in names: self.model_var.set(names[0])
        self.update_model_info()
        self.current_label.configure(text=f"Current model: {self.model_var.get()}")

    def model_changed(self, _event=None):
        self.current_label.configure(text=f"Current model: {self.model_var.get()}")
        self.put(f"Model selected: {self.model_var.get()}\n", "info")
        self.update_model_info()

    def check_runtime_periodically(self):
        def check():
            try: installed_models(); ok = True
            except Exception: ok = False
            self.q.put(("runtime", ok))
        if not self.busy: threading.Thread(target=check, daemon=True).start()
        self.root.after(10000, self.check_runtime_periodically)

    def on_enter(self, event):
        if event.state & 0x0001: return None
        self.send(); return "break"

    def send(self):
        if self.busy: return
        prompt = self.get_prompt()
        if not prompt:
            messagebox.showwarning("Empty prompt", "Enter a question before sending."); return
        model = self.model_var.get(); style = self.style_var.get()
        self.hide_placeholder(); self.prompt.delete("1.0", "end")
        self.put(f"You · {datetime.now().strftime('%H:%M')}\n", "label"); self.put(prompt + "\n", "user")
        self.messages.append({"role":"user", "content":prompt})
        outgoing = self.messages[-CONTEXT_MESSAGES:]
        while outgoing and outgoing[0].get("role") != "user": outgoing = outgoing[1:]
        selected_style = STYLES.get(style, "")
        system = CHAT_LENGTH_INSTRUCTION + ("\n\n" + selected_style if selected_style else "")
        outgoing = [{"role":"system", "content":system}] + outgoing
        self.stop_event = threading.Event(); self.busy = True; self.ai_started = False
        self.send_btn.pack_forget(); self.stop_btn.pack(fill="both", expand=True); self.model_box.configure(state="disabled"); self.style_box.configure(state="disabled")
        threading.Thread(target=self.chat_worker, args=(model, outgoing, prompt, style, self.stop_event), daemon=True).start()

    def chat_worker(self, model, messages, prompt, style, stop_event):
        monitor = ResourceMonitor(); monitor.start()
        try:
            answer, data, elapsed, first = chat_request(model, messages, stop_event, lambda chunk: self.q.put(("chunk", chunk)), stream=True)
            err = ""
        except Exception as exc:
            answer = ""; data = {}; elapsed = 0; first = None; err = friendly_error(exc, model)
        finally:
            monitor.stop()
        self.q.put(("chat_done", model, prompt, style, answer, data, elapsed, first, err, stop_event.is_set(), monitor.summary()))

    def stop(self):
        if self.busy:
            self.stop_event.set(); self.stats.configure(text="Stopping after the current stream chunk…")

    def finish_busy(self):
        self.busy = False; self.stop_btn.pack_forget(); self.send_btn.pack(fill="both", expand=True)
        self.model_box.configure(state="readonly"); self.style_box.configure(state="readonly")

    def render_reply(self, text):
        # Small formatter for bold and inline-code markers; all other text is preserved.
        parts = re.split(r"(\*\*.*?\*\*|`[^`]+`)", text, flags=re.S)
        for part in parts:
            if part.startswith("**") and part.endswith("**") and len(part) > 4: self.put(part[2:-2], "bold")
            elif part.startswith("`") and part.endswith("`") and len(part) > 2: self.put(part[1:-1], "code")
            else: self.put(part, "assistant")
        self.put("\n", "assistant")

    def on_chat_done(self, model, prompt, style, answer, data, elapsed, first, err, stopped, resources):
        self.finish_busy()
        if err:
            if self.messages and self.messages[-1].get("role") == "user": self.messages.pop()
            self.put("Error: " + err + "\n", "error"); return
        if not answer.strip():
            if stopped:
                if self.messages and self.messages[-1].get("role") == "user": self.messages.pop()
                self.put("[Response stopped.]\n", "info"); return
            answer = "(The model returned an empty response.)"
        if not self.ai_started:
            self.put(f"{model} · {datetime.now().strftime('%H:%M')}\n", "label")
            self.render_reply(answer)
        else:
            # Replace the streamed plain text with the final formatted response.
            try:
                self.chat.configure(state="normal")
                self.chat.delete("reply_start", "end-1c")
                self.chat.configure(state="disabled")
                self.put(f"{model} · {datetime.now().strftime('%H:%M')}\n", "label")
                self.render_reply(answer)
            except tk.TclError:
                self.put(answer + "\n", "assistant")
        if stopped: self.put("[Response stopped.]\n", "info")
        self.messages.append({"role":"assistant", "content":answer}); self.last_reply = answer
        stats = api_stats(data)
        speed = "—"
        try:
            if data.get("eval_count") and data.get("eval_duration"): speed = f"{data['eval_count'] / (data['eval_duration']/1e9):.1f} tokens/s"
        except Exception: pass
        self.stats.configure(text=f"Response: {elapsed:.2f} s   First token: {f'{first:.2f} s' if first is not None else '—'}   Speed: {speed}   Model load: {stats['load_duration_seconds']} s")
        self.times.append(elapsed)
        entry = {"timestamp":datetime.now().isoformat(timespec="seconds"), "model":model, "style":style, "prompt":prompt, "response":answer, "response_time_sec":round(elapsed, 4), "first_token_sec":round(first,4) if first is not None else "", "load_time_sec":stats["load_duration_seconds"], "tokens_per_sec":stats["tokens_per_second"], "eval_count":stats["eval_count"], "stopped":stopped}
        self.history.append(entry)
        if not save_history(self.history): self.put("Warning: history could not be saved. Check folder permissions.\n", "error")
        try:
            log_conversation_csv(entry, model_details(model))
        except Exception as exc:
            self.put(f"Warning: conversation CSV could not be updated ({exc}). Close it if it is open in Excel.\n", "error")

    def update_model_info(self):
        model = self.model_var.get()
        self.info_label.configure(text=f"Model details: reading details for {model}…")
        threading.Thread(target=lambda: self.q.put(("modelinfo", model, model_details(model))), daemon=True).start()

    def show_model_info(self, model, info):
        if model != self.model_var.get():
            return
        parts = [("Parameters", info.get("parameter_size")), ("Quantization", info.get("quantization")),
                 ("File size", info.get("file_size")), ("Context", info.get("context_length")),
                 ("Family", info.get("family")), ("Format", info.get("format"))]
        text = " | ".join(f"{k}: {v}" for k, v in parts if v) or "unavailable (is Ollama running?)"
        self.info_label.configure(text=f"Model details: {text} | Source: {info.get('source', '')}")

    def export_model_info(self):
        names = list(self.model_box["values"])
        def work():
            try:
                self.q.put(("info", f"Model details saved to {save_model_info(names)}\n"))
            except Exception as exc:
                self.q.put(("warn", f"Could not save model info: {exc}\n"))
        threading.Thread(target=work, daemon=True).start()

    def summarize_results(self, quiet=False):
        try:
            self.put(write_summary() + "\n", "info")
        except Exception as exc:
            if quiet:
                self.put(f"Summary not written: {exc}\n", "error")
            else:
                messagebox.showerror("Summary", str(exc))

    def new_conversation(self):
        if self.busy: return
        self.messages = []; self.last_reply = ""; self.clear_screen(); self.put("New conversation started. Saved history is preserved.\n", "info")

    def clear_history(self):
        if self.busy: return
        if not messagebox.askyesno("Clear history", "Delete the saved history and clear the current conversation?"): return
        self.history = []; self.messages = []; self.last_reply = ""
        if not save_history(self.history): messagebox.showerror("History", "Could not write the history file.")
        self.clear_screen(); self.put("Saved history and current conversation cleared.\n", "info")

    def copy_reply(self):
        if not self.last_reply: messagebox.showinfo("Copy reply", "There is no reply to copy yet."); return
        self.root.clipboard_clear(); self.root.clipboard_append(self.last_reply)

    def view_history(self):
        win = tk.Toplevel(self.root); win.title("Recent conversation history"); win.geometry("800x550"); win.configure(bg=self.P["bg"])
        box = scrolledtext.ScrolledText(win, wrap="word", font=(FONT,10), bg=self.P["panel"], fg=self.P["text"]); box.pack(fill="both", expand=True, padx=10, pady=10)
        for i, e in enumerate(self.history[-20:], 1):
            box.insert("end", f"#{i} | {e.get('timestamp','')} | {e.get('model','')} | {e.get('response_time_sec','')} s | {e.get('style','')}\n")
            box.insert("end", f"You: {e.get('prompt','')}\nAssistant: {e.get('response','')}\n{'─'*65}\n")
        box.configure(state="disabled")

    def export_history(self):
        if not self.history: messagebox.showinfo("Export history", "There is no saved history yet."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="conversation_history.csv", filetypes=[("CSV files","*.csv")])
        if not path: return
        columns = ["timestamp","model","style","prompt","response","response_time_sec","first_token_sec","load_time_sec","tokens_per_sec","stopped"]
        try:
            with open(path,"w",newline="",encoding="utf-8-sig") as f:
                w=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore"); w.writeheader(); w.writerows(self.history)
            messagebox.showinfo("Export history", f"Saved {len(self.history)} entries to:\n{path}")
        except OSError as exc: messagebox.showerror("Export history", str(exc))

    def refresh_csv_status(self):
        """Show how many benchmark records are currently saved on disk."""
        try:
            if not BENCHMARK_FILE.exists() or BENCHMARK_FILE.stat().st_size == 0:
                self.csv_status.configure(text="Benchmark results: no scored rows yet")
                return
            with BENCHMARK_FILE.open("r", newline="", encoding="utf-8-sig") as handle:
                count = max(0, sum(1 for _ in csv.DictReader(handle)))
            self.csv_status.configure(text=f"Benchmark results saved: {count} row(s)")
        except (OSError, csv.Error):
            self.csv_status.configure(text="Benchmark CSV: unable to read")

    def view_benchmark_csv(self):
        """Open a readable preview of saved benchmark and model-load CSV rows."""
        win = tk.Toplevel(self.root)
        win.title("Saved benchmark results")
        win.geometry("1100x650")
        win.configure(bg=self.P["bg"])
        top = tk.Frame(win, bg=self.P["panel"], padx=10, pady=8)
        top.pack(fill="x")
        tk.Label(top, text=f"Benchmark CSV: {BENCHMARK_FILE}", bg=self.P["panel"], fg=self.P["text"], anchor="w", justify="left", wraplength=950).pack(fill="x")
        tk.Label(top, text=f"Load-test CSV: {LOAD_FILE}", bg=self.P["panel"], fg=self.P["muted"], anchor="w", justify="left", wraplength=950).pack(fill="x", pady=(4,0))
        controls = tk.Frame(win, bg=self.P["bg"], padx=10, pady=6); controls.pack(fill="x")
        selected_file = tk.StringVar(value="Benchmark results")
        selector = ttk.Combobox(controls, textvariable=selected_file, values=["Benchmark results", "Model-load tests"], state="readonly", width=22)
        selector.pack(side="left", padx=(0,8))
        preview = scrolledtext.ScrolledText(win, wrap="none", font=(CODE_FONT, 9), bg=self.P["panel"], fg=self.P["text"])
        preview.pack(fill="both", expand=True, padx=10, pady=(0,10))

        def load_preview():
            path = BENCHMARK_FILE if selected_file.get() == "Benchmark results" else LOAD_FILE
            preview.configure(state="normal"); preview.delete("1.0", "end")
            try:
                if not path.exists():
                    preview.insert("end", "This CSV has not been created yet. Run a benchmark first.\n")
                else:
                    with path.open("r", newline="", encoding="utf-8-sig") as handle:
                        reader = csv.DictReader(handle)
                        rows = list(reader)
                    preview.insert("end", f"File: {path}\nSaved data rows: {len(rows)}\n\n")
                    if not rows:
                        preview.insert("end", "The file exists but has no data rows yet.\n")
                    else:
                        # Show key columns first so the result is readable without
                        # forcing the user to scroll across every resource column.
                        preferred = ["timestamp", "trial_id", "trial_number", "model", "prompt_id", "prompt_category", "response_time_seconds", "system_cpu_avg_percent", "system_cpu_max_percent", "ollama_ram_avg_mb", "ollama_ram_max_mb", "status", "answer", "error"]
                        columns = [key for key in preferred if key in reader.fieldnames]
                        for row_number, row in enumerate(rows[-100:], start=max(1, len(rows)-99)):
                            preview.insert("end", f"--- Saved row {row_number} ---\n")
                            for key in columns:
                                value = (row.get(key) or "").replace("\r", " ").replace("\n", " ")
                                if len(value) > 500: value = value[:500] + " … [trimmed in preview; full answer remains in CSV]"
                                preview.insert("end", f"{key}: {value}\n")
                            preview.insert("end", "\n")
                        if len(rows) > 100:
                            preview.insert("end", "Preview shows only the latest 100 rows. The CSV file contains all rows.\n")
            except (OSError, csv.Error, UnicodeError) as exc:
                preview.insert("end", f"Could not read CSV: {exc}\n")
            preview.configure(state="disabled")
            self.refresh_csv_status()

        ttk.Button(controls, text="Refresh preview", command=load_preview).pack(side="left", padx=4)
        ttk.Button(controls, text="Open CSV in spreadsheet app", command=lambda: self.open_csv_file(BENCHMARK_FILE if selected_file.get() == "Benchmark results" else LOAD_FILE)).pack(side="left", padx=4)
        selector.bind("<<ComboboxSelected>>", lambda _event: load_preview())
        load_preview()

    def open_csv_file(self, path):
        """Ask Windows/macOS/Linux to open a CSV with the default spreadsheet app."""
        if not path.exists():
            messagebox.showinfo("CSV not found", f"No file exists yet:\n{path}\n\nRun a benchmark first.", parent=self.root)
            return
        try:
            if os.name == "nt":
                os.startfile(str(path))
            elif shutil.which("open"):
                subprocess.Popen(["open", str(path)])
            elif shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", str(path)])
            else:
                messagebox.showinfo("CSV location", str(path), parent=self.root)
        except Exception as exc:
            messagebox.showerror("Open CSV", f"Could not open the CSV automatically.\n{path}\n\n{exc}", parent=self.root)

    def open_results_folder(self):
        """Open the folder containing the two benchmark CSV files."""
        try:
            if os.name == "nt":
                os.startfile(str(BASE_DIR))
            elif shutil.which("open"):
                subprocess.Popen(["open", str(BASE_DIR)])
            elif shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", str(BASE_DIR)])
            else:
                messagebox.showinfo("Results folder", str(BASE_DIR), parent=self.root)
        except Exception as exc:
            messagebox.showerror("Results folder", f"Could not open the folder automatically.\n{BASE_DIR}\n\n{exc}", parent=self.root)

    def start_benchmark(self):
        if self.busy:
            messagebox.showinfo("Busy", "Wait for the current response to finish first."); return
        try:
            archived = [p for p in (ensure_csv(BENCHMARK_FILE, BENCH_COLUMNS), ensure_csv(LOAD_FILE, LOAD_COLUMNS)) if p]
            for p in archived: self.put(f"Old CSV had a different layout and was archived as {p.name}\n", "info")
        except Exception as exc:
            messagebox.showerror("CSV setup", str(exc)); return

        # Let the user choose one or both installed models.
        available = list(self.model_box["values"])
        if not available:
            messagebox.showerror("No models", "No Ollama models are available. Start Ollama and refresh models first."); return
        model_mode = simpledialog.askstring(
            "Benchmark models",
            "Choose model run mode:\n1 = Current selected model only\n2 = All installed models, one after another\n\nThe same selected prompts and trial count will be used for each model.\nEnter 1 or 2:",
            initialvalue="1", parent=self.root)
        if model_mode not in ("1", "2"): return
        models = [self.model_var.get()] if model_mode == "1" else available

        runs = simpledialog.askinteger("Benchmark trials", "How many trials per prompt? (Use at least 3 for final results):", initialvalue=3, minvalue=1, maxvalue=20, parent=self.root)
        if not runs: return

        prompt_mode = simpledialog.askstring(
            "Benchmark prompts",
            "Choose fixed-prompt mode:\n1 = Run one prompt separately (choose P01–P10 next)\n2 = Run all 10 fixed prompts automatically\n\nEnter 1 or 2:",
            initialvalue="2", parent=self.root)
        if prompt_mode not in ("1", "2"): return
        if prompt_mode == "1":
            menu = "\n".join(f"{pid} — {category}: {prompt}" for pid, category, prompt in PROMPTS)
            selected = simpledialog.askstring("Select fixed prompt", menu + "\n\nEnter prompt ID (P01–P10):", initialvalue="P01", parent=self.root)
            if not selected: return
            selected = selected.strip().upper()
            if selected not in {item[0] for item in PROMPTS}:
                messagebox.showerror("Invalid prompt", "Enter a prompt ID from P01 to P10."); return
            prompt_ids = [selected]
        else:
            prompt_ids = [item[0] for item in PROMPTS]

        total = len(models) * len(prompt_ids) * runs
        model_list = ", ".join(models)
        if not messagebox.askyesno("Confirm benchmark", f"Models: {model_list}\nPrompts: {len(prompt_ids)}\nTrials per prompt per model: {runs}\nTotal scored responses: {total}\n\nThe benchmark ignores chat style so models are compared fairly. Start?", parent=self.root): return
        self.stop_event = threading.Event(); self.busy = True; self.benchmarking = True; self.send_btn.pack_forget(); self.stop_btn.pack(fill="both", expand=True); self.model_box.configure(state="disabled"); self.style_box.configure(state="disabled")
        self.put(f"\nBenchmark started: {model_list}; {len(prompt_ids)} prompt(s) × {runs} trial(s) per model.\n", "info")
        threading.Thread(target=self.benchmark_worker, args=(models, prompt_ids, runs, self.stop_event), daemon=True).start()

    def benchmark_worker(self, models, prompt_ids, runs, stop_event):
        try:
            completed=0; times=[]; errors=0; total=len(models)*len(prompt_ids)*runs
            for model in models:
                info = model_details(model, refresh=True)
                try:
                    if shutil.which("ollama"):
                        subprocess.run(["ollama","stop",model],capture_output=True,text=True,timeout=30); time.sleep(2)
                except Exception: pass
                loadrow = {"timestamp":datetime.now().isoformat(timespec="seconds"),"model":model,"model_file_size":info.get("file_size",""),"parameter_size":info.get("parameter_size",""),"quantization":info.get("quantization",""),"load_test_type":"stop_then_first_request","status":"success","error":""}
                try:
                    _, data, elapsed, _ = chat_request(model,[{"role":"user","content":"Reply with exactly: load test complete"}],stream=False)
                    loadrow.update({"response_time_seconds":round(elapsed,4), **api_stats(data)})
                except Exception as exc:
                    loadrow["status"]="error"; loadrow["error"]=friendly_error(exc,model)
                append_csv(LOAD_FILE,LOAD_COLUMNS,loadrow)
                if loadrow["status"] != "success":
                    self.q.put(("bench_progress", completed, total, model, 0.0, "error")); errors += 1
                    continue
                try:
                    chat_request(model,[{"role":"user","content":"Reply with exactly: warm-up complete"}],stream=False)
                except Exception as exc:
                    self.q.put(("bench_progress", completed, total, model, 0.0, "error")); errors += 1
                    continue
                for trial in range(1,runs+1):
                    for pid in prompt_ids:
                        if stop_event.is_set():
                            self.q.put(("bench_done",model,completed,total,"Benchmark stopped by user.",sum(times)/len(times) if times else None,min(times) if times else None,max(times) if times else None,errors)); return
                        _, category, prompt = next(item for item in PROMPTS if item[0] == pid)
                        monitor=ResourceMonitor(); monitor.start(); start=time.perf_counter()
                        try:
                            answer,data,elapsed,_=chat_request(model,[{"role":"user","content":prompt}],stop_event=stop_event,stream=False)
                            status="success"; error=""; times.append(elapsed)
                        except Exception as exc:
                            answer=""; data={}; elapsed=round(time.perf_counter()-start,4); status="error"; error=friendly_error(exc,model); errors+=1
                        finally: monitor.stop()
                        ps = loaded_info(model) if status == "success" else {}
                        row={"timestamp":datetime.now().isoformat(timespec="seconds"),"trial_id":f"{model}_{pid}_T{trial}_{datetime.now().strftime('%Y%m%d_%H%M%S')}","trial_number":trial,"model":model,"model_file_size":info.get("file_size",""),"parameter_size":info.get("parameter_size",""),"quantization":info.get("quantization",""),"model_family":info.get("family",""),"model_format":info.get("format",""),"context_length":info.get("context_length",""),"prompt_id":pid,"prompt_category":category,"prompt":prompt,"generation_temperature":GENERATION_OPTIONS["temperature"],"generation_num_predict":GENERATION_OPTIONS["num_predict"],"response_time_seconds":elapsed,**(api_stats(data) if data else {}),**monitor.summary(),**ps,"status":status,"answer":answer,"error":error,"correctness_score":"","relevance_score":"","completeness_score":"","evaluation_notes":""}
                        saved=append_csv(BENCHMARK_FILE,BENCH_COLUMNS,row)
                        if saved != BENCHMARK_FILE: self.q.put(("warn", f"{BENCHMARK_FILE.name} is locked (close it in Excel). Row kept in {saved.name}.\n"))
                        completed+=1
                        self.q.put(("bench_progress",completed,total,f"{model} {pid} T{trial}",elapsed,status))
            self.q.put(("bench_done",", ".join(models),completed,total,"",sum(times)/len(times) if times else None,min(times) if times else None,max(times) if times else None,errors))
        except Exception as exc:
            self.q.put(("bench_done",", ".join(models),0,total,f"Benchmark failed: {type(exc).__name__}: {exc}"))

    def poll_queue(self):
        try:
            while True:
                item=self.q.get_nowait(); kind=item[0]
                if kind=="models": self.apply_models(item[1],item[2])
                elif kind=="runtime": self.runtime_label.configure(text="Ollama: running" if item[1] else "Ollama: not running")
                elif kind=="chunk":
                    if not self.ai_started:
                        self.ai_started=True
                        self.chat.mark_set("reply_start", "end-1c")
                        self.chat.mark_gravity("reply_start", "left")
                        self.put(f"{self.model_var.get()} · {datetime.now().strftime('%H:%M')}\n","label")
                    self.put(item[1],"assistant")
                elif kind=="chat_done": self.on_chat_done(*item[1:])
                elif kind=="modelinfo": self.show_model_info(item[1], item[2])
                elif kind=="info": self.put(item[1], "info")
                elif kind=="warn": self.put(item[1], "error")
                elif kind=="bench_progress":
                    self.put(f"Benchmark {item[1]}/{item[2]} · {item[3]} · {item[4]:.2f} s · {item[5]}\n","info")
                    self.refresh_csv_status()
                elif kind=="bench_done":
                    self.finish_busy(); self.benchmarking=False; self.summarize_results(quiet=True)
                    model,done,total,msg,*rest=item[1:]
                    self.put(f"Benchmark finished for {model}: {done}/{total} trials recorded.\n", "info")
                    if msg: self.put(msg+"\n","error" if "failed" in msg.lower() or "error" in msg.lower() else "info")
                    if rest and rest[0] is not None:
                        avg,fast,slow,errors=rest
                        self.put(f"Mean response time: {avg:.2f} s | Fastest: {fast:.2f} s | Slowest: {slow:.2f} s | Errors: {errors}\n", "info")
                    self.put(f"Benchmark CSV: {BENCHMARK_FILE}\nLoad-test CSV: {LOAD_FILE}\nUse ‘View benchmark CSV’ to preview saved rows or ‘Open results folder’ to open the files.\n","info")
                    self.refresh_csv_status()
        except queue.Empty: pass
        self.root.after(40,self.poll_queue)

    def update_live_resources(self):
        """Display live whole-system CPU/RAM and detected Ollama process usage."""
        if not psutil:
            self.live_resources.configure(
                text="Live resources unavailable: install psutil with  python -m pip install psutil"
            )
            self.root.after(2000, self.update_live_resources)
            return
        try:
            system_cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            logical_cpus = psutil.cpu_count(logical=True) or 1
            active = set()
            cpu_total = 0.0
            ram_total = 0
            for proc in psutil.process_iter(["pid", "name", "exe"]):
                try:
                    info = proc.info
                    identity = ((info.get("name") or "") + " " + (info.get("exe") or "")).lower()
                    if "ollama" not in identity and "llama-server" not in identity:
                        continue
                    pid = proc.pid
                    active.add(pid)
                    # First sample primes psutil; later samples provide a meaningful CPU percentage.
                    if pid not in self.live_processes:
                        proc.cpu_percent(interval=None)
                        self.live_processes[pid] = proc
                    else:
                        tracked = self.live_processes[pid]
                        cpu_total += tracked.cpu_percent(interval=None)
                        ram_total += tracked.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            self.live_processes = {pid: proc for pid, proc in self.live_processes.items() if pid in active}
            ollama_cpu = cpu_total / logical_cpus
            ollama_ram_mb = ram_total / (1024 * 1024)
            runtime_text = "Ollama process detected" if active else "No Ollama process detected yet"
            self.live_resources.configure(
                text=(f"LIVE RESOURCES  |  System CPU: {system_cpu:.1f}%  |  "
                      f"System RAM: {memory.percent:.1f}% used ({memory.used / (1024**3):.1f}/"
                      f"{memory.total / (1024**3):.1f} GB)  |  "
                      f"Ollama CPU: {ollama_cpu:.1f}% of total CPU capacity  |  "
                      f"Ollama RAM: {ollama_ram_mb:.0f} MB  |  {runtime_text}")
            )
        except Exception as exc:
            self.live_resources.configure(text=f"Live resource monitor error: {exc}")
        self.root.after(1500, self.update_live_resources)

    def animate(self):
        if self.busy:
            if self.benchmarking: self.stats.configure(text="Benchmark running… (press Stop to request cancellation)")
            else: self.stats.configure(text="Generating local response… (press Stop to cancel)")
        self.root.after(500,self.animate)

    def close(self):
        if self.busy and not messagebox.askyesno("Exit", "A request is still running. Exit the application?"): return
        self.stop_event.set(); self.root.destroy()


def main():
    root=tk.Tk(); App(root); root.mainloop()

if __name__ == "__main__":
    main()
