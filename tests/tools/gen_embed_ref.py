#!/usr/bin/env python3
"""Write llama.cpp reference embeddings for tests/test_embed_matrix.c.

  gen_embed_ref.py <llama-embedding> <models_dir> <out_dir>      run llama.cpp
  gen_embed_ref.py --from-json ref.json <out_dir>                convert a saved run

Output per model: <stem>.ref = 'ERF1', int32 dim, int32 n_short, int32 has_long,
then (n_short + has_long) L2-normalised float32 vectors. The texts must match
SHORT[] / long_text() in test_embed_matrix.c.
"""
import json, os, struct, subprocess, sys

SHORT = ["The quick brown fox jumps over the lazy dog.",
         "A fast brown fox leaps over a sleepy dog.",
         "Quarterly revenue grew by twelve percent in the European market.",
         "dog lazy the over jumps fox brown quick The.",
         "def add(a, b):\n    return a + b",
         "Türkiye'nin başkenti Ankara'dır; İstanbul çok büyük bir şehirdir. Ünlü şair Yunus Emre'yi okudum.",
         "日本語のテキストを埋め込みます。Café naïve résumé 123456 !!! 東京タワー"]
BASE = ["The committee reviewed the annual budget and approved several infrastructure projects across the region.",
        "Engineers measured the vibration of the bridge under heavy traffic and recorded the results in a report.",
        "A small bakery on the corner of the street sells fresh bread, croissants and coffee every morning.",
        "Researchers trained a neural network to classify satellite images of forests, rivers and cities.",
        "The museum opened a new exhibition about ancient trade routes connecting Asia, Africa and Europe."]
LONG = " ".join("%s (%d)" % (BASE[i % 5], i) for i in range(75))
SEP = "<#sep#>"


def run(exe, model, texts, extra):
    cmd = [exe, "-m", model, "--embd-output-format", "json", "--embd-normalize", "2", "-t", "4",
           "--embd-separator", SEP, "-p", SEP.join(texts)] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    try:
        return [d["embedding"] for d in json.loads(r.stdout)["data"]]
    except Exception:
        return None


def write(out_dir, name, short, long):
    dim = len(short[0])
    with open(os.path.join(out_dir, os.path.splitext(name)[0] + ".ref"), "wb") as f:
        f.write(b"ERF1" + struct.pack("<iii", dim, len(short), 1 if long else 0))
        for v in short + (long or []):
            f.write(struct.pack("<%df" % dim, *v))


if sys.argv[1] == "--from-json":
    data, out = json.load(open(sys.argv[2])), sys.argv[3]
    for name, r in data.items():
        write(out, name, r["short"], r["long"])
else:
    exe, models, out = sys.argv[1:4]
    for name in sorted(os.listdir(models)):
        if name.endswith(".gguf"):
            p = os.path.join(models, name)
            short = run(exe, p, SHORT, [])
            long = run(exe, p, [LONG], ["-c", "4096", "-b", "4096", "-ub", "4096"])
            if short:
                write(out, name, short, long)
