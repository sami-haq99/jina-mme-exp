"""
app.py – Token‑wise heatmaps with JinaV4SimilarityMapper (similarity4)

───────── Feature checklist (for future edits) ─────────
1. Prompt + Image‑URL inputs (placeholders).                       ✅
2. Run → downloads image (≥512 h), gets tokens+heatmaps,           ✅
   auto‑selects first token, shows overlay, clears inputs.
3. Output widgets hidden until results are ready.                  ✅
4. Every run saved to examples/auto_<timestamp>/ with:
     • prompt.txt, img_url.txt, image.jpg,
     • heatmaps.json, per‑token PNGs, preview_first_token.jpg.     ✅
5. On startup, first 3 example folders rendered below output with
   layout: Prompt → Image URL → Tokens → Image+Heatmap.            ✅
6. Margins: 40 px before “Examples” heading, 25 px between         ✅
   successive examples (no separators, no extra HTML).
7. Works on gradio==5.35.0 (no gr.Box, no Button.style, etc.).     ✅
"""

import sys, signal, base64, re, io, json, time
from io import BytesIO
from pathlib import Path
from typing import Dict
import subprocess

import requests
import gradio as gr
from PIL import Image
from similarity import JinaV4SimilarityMapper

EX_DIR = Path("examples"); EX_DIR.mkdir(exist_ok=True)


ButtonsLike = gr.Radio
def buttons_update(toks):
    first = toks[0] if toks else None
    return gr.update(choices=toks, value=first, visible=True)

# ───────── util functions ─────────
def _slug(t: str, n: int = 60) -> str:
    return re.sub(r"[^\w\-]+", "_", t.lower())[:n] or "x"

def overlay(tok: str, maps: Dict[str, str], base: Image.Image) -> Image.Image:
    if tok not in maps:
        return base
    hm = Image.open(BytesIO(base64.b64decode(maps[tok]))).convert("RGBA")
    if hm.size != base.size:
        hm = hm.resize(base.size, Image.BILINEAR)
    return Image.alpha_composite(base.convert("RGBA"), hm)

def save_run(prompt: str, url: str, img: Image.Image, maps: Dict[str, str]) -> None:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    fldr = EX_DIR / f"auto_{_slug(prompt,30)}_{ts}"
    fldr.mkdir(parents=True, exist_ok=True)

    (fldr / "prompt.txt").write_text(prompt)
    (fldr / "img_url.txt").write_text(url)
    img.convert("RGB").save(fldr / "image.jpg", "JPEG")

    with (fldr / "heatmaps.json").open("w") as f:
        json.dump(maps, f)

    for tok, b64png in maps.items():
        (fldr / f"heatmap_{_slug(tok,30)}.png").write_bytes(base64.b64decode(b64png))

    first = next(iter(maps))
    overlay(first, maps, img).convert("RGB").save(fldr / "preview_first_token.jpg", "JPEG")
    print(f"✨ Saved run to {fldr}", flush=True)

# ───────── load mapper ─────────
print("⏳ Loading JinaV4SimilarityMapper …", flush=True)
MAPPER = JinaV4SimilarityMapper(client_type="web")
print("✅ Mapper ready.", flush=True)

# ───────── load up to 3 example folders ─────────
def load_examples(n: int = 3):
    ex = []
    for fld in sorted(EX_DIR.iterdir())[:n]:
        p_txt, p_url, p_img, p_map = fld/"prompt.txt", fld/"img_url.txt", None, fld/"heatmaps.json"
        for c in fld.glob("image.*"): p_img = c; break
        if not (p_txt.exists() and p_url.exists() and p_img and p_map.exists()): continue
        ex.append(dict(
            prompt=p_txt.read_text().strip(),
            url   =p_url.read_text().strip(),
            base  =Image.open(p_img).convert("RGB"),
            maps  =json.load(open(p_map))
        ))
    return ex

static_examples = load_examples()

# ───────── backend for user Run ─────────
def run_mapper(prompt: str, img_url: str, api_key: str):
    new_client = JinaV4SimilarityMapper(client_type="web")
    if not img_url:
        raise gr.Error("Please provide an image URL.")
    if not prompt:
        raise gr.Error("Please provide a prompt.")
    if not api_key:
        raise gr.Error("Please provide a valid API key.")
    try:
        r = requests.get(img_url, stream=True, timeout=10); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise gr.Error(f"Image load failed: {e}")
    new_client.model.set_api_key(api_key)
    img_proc, *_ =  new_client.process_image(img_url)
    toks, maps = new_client.get_token_similarity_maps(prompt, img_proc)
    if not toks:
        raise gr.Error("Mapper returned no tokens.")

    # save_run(prompt, img_url, img_proc, maps)

    first_tok = toks[0]
    info      = f"**Prompt:** {prompt}\n\n**Image URL:** {img_url}"
    return (
        buttons_update(toks), maps, img_proc,
        gr.update(value=overlay(first_tok, maps, img_proc), visible=True),
        gr.update(value=info, visible=True),
        "", "")

# ───────── UI ─────────
css = """
#main-title { margin-bottom: 40px; }
#run-btn { margin: 20px 0; }
#examples-title { margin: 40px 0; }
.example-space { margin: 20px 0; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Jina Embeddings v4", elem_id="main-title")

    # User input
    prompt_in = gr.Textbox(label="Prompt", placeholder="Describe what to query…")
    url_in    = gr.Textbox(label="Image URL", placeholder="https://example.com/image.jpg")
    api_key_in = gr.Textbox(label="API Key", placeholder="Enter your Jina API key here")
    run_btn   = gr.Button("Run", elem_id="run-btn")

    # Output area
    info_md   = gr.Markdown(visible=False)
    token_sel = ButtonsLike(choices=[], label="Tokens", interactive=True, visible=False)
    maps_st   = gr.State({})
    img_st    = gr.State(None)
    img_out   = gr.Image(label="Image + Heatmap", visible=False)

    run_btn.click(run_mapper,
        [prompt_in, url_in, api_key_in],
        [token_sel, maps_st, img_st, img_out, info_md, prompt_in, url_in])

    (token_sel.select if hasattr(token_sel,"select") else token_sel.change)(
        overlay, [token_sel, maps_st, img_st], [img_out])

    # Margin before examples heading
    gr.Markdown("## Examples", elem_id="examples-title")

    # Render examples
    for ex in static_examples:
        gr.Markdown(f"**Prompt:** {ex['prompt']}")
        gr.Markdown(f"**Image URL:** {ex['url']}")
        ex_img_st  = gr.State(ex["base"])
        ex_map_st  = gr.State(ex["maps"])
        first      = next(iter(ex["maps"]))
        ex_btns    = ButtonsLike(choices=list(ex["maps"].keys()), value=first, interactive=True)
        ex_disp    = gr.Image(value=overlay(first, ex["maps"], ex["base"]))
        (ex_btns.select if hasattr(ex_btns,"select") else ex_btns.change)(
            overlay, [ex_btns, ex_map_st, ex_img_st], [ex_disp])
        # vertical margin after each example
        gr.Markdown("", elem_classes=["example-space"])

# ───────── graceful shutdown ─────────
def _shutdown(*_): print("🛑 Shutting down …", flush=True); demo.close(); sys.exit(0)
signal.signal(signal.SIGINT, _shutdown); signal.signal(signal.SIGTERM, _shutdown)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=False)
