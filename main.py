import base64, io, os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image

# Safer defaults for CPU
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()
HF_MODEL_IMAGE = os.getenv("HF_MODEL_IMAGE", "stabilityai/sd-turbo")

app = FastAPI(title="vision-ai-backend-local", version="1.0.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_pipe = None  # lazy-loaded pipeline

class GenBody(BaseModel):
    prompt: str = Field(..., min_length=3)
    negative_prompt: Optional[str] = None
    steps: int = Field(4, ge=1, le=40)                # faster default
    guidance_scale: float = Field(2.5, ge=0.0, le=20.0)
    width: int = Field(352, ge=256, le=1024)          # slightly smaller than 384 for speed
    height: int = Field(352, ge=256, le=1024)
    seed: Optional[int] = None

def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _init_local():
    global _pipe
    if _pipe is not None:
        return
    import torch
    from diffusers import AutoPipelineForText2Image

    # make CPU runs more predictable
    try:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    except Exception:
        pass

    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32
    print(f"[boot] cuda={has_cuda} dtype={dtype}", flush=True)

    # IMPORTANT: no 'variant' arg when on CPU to avoid model variant mismatches
    if has_cuda:
        _pipe = AutoPipelineForText2Image.from_pretrained(
            HF_MODEL_IMAGE, torch_dtype=dtype, use_safetensors=True, variant="fp16"
        ).to("cuda")
        try:
            _pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        _pipe = AutoPipelineForText2Image.from_pretrained(
            HF_MODEL_IMAGE, torch_dtype=dtype, use_safetensors=True
        )
        _pipe.enable_attention_slicing()

@app.get("/health")
def health():
    return {"ok": True, "mode": INFERENCE_MODE, "model_image": HF_MODEL_IMAGE}

@app.post("/generate")
def generate(body: GenBody):
    if INFERENCE_MODE != "local":
        raise HTTPException(status_code=400, detail="This build runs local mode only.")
    try:
        _init_local()
        import torch
        g = None
        if body.seed is not None:
            g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(body.seed))

        img = _pipe(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            num_inference_steps=int(body.steps),
            guidance_scale=float(body.guidance_scale),
            width=int(body.width),
            height=int(body.height),
            generator=g,
        ).images[0]
        return {"ok": True, "mode": "local", "model": HF_MODEL_IMAGE, "png_base64": _to_b64(img)}
    except Exception as e:
        print(f"[gen] error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- built-in simple viewer so we don't rely on nginx ---
_VIEWER = """<!doctype html><html><head><meta charset="utf-8"><title>Vision AI - Local</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;max-width:900px}
  h1{margin:0 0 8px;font-size:24px}
  .row{display:grid;gap:8px;grid-template-columns:1fr 1fr}
  label{font-weight:600;font-size:14px}
  textarea,input{width:100%;padding:8px;font-size:14px}
  button{padding:10px 16px;font-weight:700;cursor:pointer}
  #out{margin-top:16px}
  img{max-width:100%;border-radius:12px;display:block}
  .muted{color:#666;font-size:12px}
  .grid{display:grid;gap:12px}
  .error{color:crimson;font-weight:600}
</style></head><body>
<h1>Vision AI  Local (Text  Image)</h1>
<p class="muted">Backend: <code>/generate</code> (CPU-friendly defaults)</p>
<div class="grid">
  <label>Prompt
    <textarea id="prompt" rows="3">photo of a red surfboard on wet sand at golden hour, cinematic lighting</textarea>
  </label>
  <div class="row">
    <label>Negative
      <input id="negative" value="blurry, lowres, watermark, text, logo, signature"/>
    </label>
    <label>Steps (140)
      <input id="steps" type="number" min="1" max="40" value="4"/>
    </label>
  </div>
  <div class="row">
    <label>Guidance (020)
      <input id="guidance" type="number" step="0.1" min="0" max="20" value="2.5"/>
    </label>
    <label>Seed (blank=random)
      <input id="seed" type="number" placeholder="e.g., 12345"/>
    </label>
  </div>
  <div class="row">
    <label>Width (2561024)
      <input id="width" type="number" min="256" max="1024" value="352"/>
    </label>
    <label>Height (2561024)
      <input id="height" type="number" min="256" max="1024" value="352"/>
    </label>
  </div>
  <div>
    <button id="goBtn">Generate</button>
    <span id="status" class="muted" style="margin-left:10px;"></span>
  </div>
</div>
<div id="out"></div>
<script>
const $=id=>document.getElementById(id);
const btn=$("goBtn"), statusEl=$("status"), out=$("out");
async function generate(){
  btn.disabled=true; statusEl.textContent="Generating"; out.innerHTML="";
  try{
    const body={
      prompt:$("prompt").value,
      negative_prompt:$("negative").value||null,
      steps:Number($("steps").value),
      guidance_scale:Number($("guidance").value),
      width:Number($("width").value),
      height:Number($("height").value),
    };
    const seedVal=$("seed").value.trim(); if(seedVal) body.seed=Number(seedVal);
    const r=await fetch("/generate",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});
    const j=await r.json(); if(!r.ok||!j.ok) throw new Error(j.detail||j.error||("HTTP "+r.status));
    const img=new Image(); img.src="data:image/png;base64,"+j.png_base64; out.appendChild(img);
    const a=document.createElement("a"); a.href=img.src; a.download="out.png"; a.textContent="Download PNG";
    a.style.display="inline-block"; a.style.marginTop="8px"; out.appendChild(a);
    statusEl.textContent="Done.";
  }catch(e){
    statusEl.textContent="";
    const p=document.createElement("p"); p.className="error"; p.textContent="Error: "+(e?.message||e);
    out.appendChild(p);
  }finally{ btn.disabled=false; }
}
$("goBtn").addEventListener("click", generate);
</script>
</body></html>"""

@app.get("/viewer", response_class=HTMLResponse)
def viewer():
    return HTMLResponse(content=_VIEWER, status_code=200)

