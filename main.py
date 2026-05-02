# # api.py
# import io
# import time
# import asyncio
# import logging
# from concurrent.futures import ThreadPoolExecutor
# from typing import Optional

# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse, Response

# from PIL import Image
# import torch
# from diffusers import Flux2KleinPipeline

# # ---------- logging ----------
# logging.basicConfig(
#     level=logging.WARNING,
#     format="%(asctime)s %(levelname)s: %(message)s",
# )

# logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("diffusers").setLevel(logging.ERROR)

# logger = logging.getLogger("api")

# # ---------- app ----------
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# torch.backends.cudnn.benchmark = True

# MODEL_PATH = "/workspace/flux"

# # ---------- model load ----------
# logger.warning("Loading model...")

# pipe = Flux2KleinPipeline.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.float16,
# )

# device = "cuda" if torch.cuda.is_available() else "cpu"
# pipe.to(device)

# logger.warning(f"Model ready on {device}")

# # ---------- warmup ----------
# if device == "cuda":
#     try:
#         logger.warning("Warmup run...")
#         dummy = Image.new("RGB", (512, 512), "white")

#         with torch.inference_mode():
#             pipe(
#                 image=dummy,
#                 prompt="test",
#                 num_inference_steps=1,
#                 height=512,
#                 width=512,
#             )

#         logger.warning("Warmup done")
#     except Exception as e:
#         logger.warning(f"Warmup failed: {e}")

# # ---------- executor ----------
# executor = ThreadPoolExecutor(max_workers=1)

# # ---------- inference ----------
# def run_inference(image: Image.Image, prompt: str, h: int, w: int):
#     start = time.time()

#     # resize large inputs (major speed gain)
#     image.thumbnail((1600, 1600))

#     with torch.inference_mode():
#         result = pipe(
#             image=image,
#             prompt=prompt,
#             num_inference_steps=6,
#             guidance_scale=1.0,
#             height=h,
#             width=w,
#         ).images[0]

#     buf = io.BytesIO()
#     result.save(buf, format="PNG", optimize=True)
#     buf.seek(0)

#     logger.warning(f"Done in {round(time.time()-start,2)}s")
#     return buf

# # ---------- routes ----------
# @app.get("/")
# def root():
#     return {"status": "api is running"}

# @app.get("/health")
# def health():
#     return {"ok": True}

# @app.get("/favicon.ico")
# def favicon():
#     return Response(status_code=204)

# @app.post("/edit")
# async def edit(
#     file: UploadFile = File(...),
#     prompt: str = Form(...),
#     height: Optional[int] = Form(768),
#     width: Optional[int] = Form(1024),
# ):
#     if not file.content_type.startswith("image"):
#         raise HTTPException(400, "Invalid image")

#     data = await file.read()

#     try:
#         image = Image.open(io.BytesIO(data)).convert("RGB")
#     except:
#         raise HTTPException(400, "Bad image")

#     loop = asyncio.get_running_loop()
#     buf = await loop.run_in_executor(
#         executor,
#         run_inference,
#         image,
#         prompt,
#         height,
#         width,
#     )

#     return StreamingResponse(buf, media_type="image/png")
# New Code Using U2Net

# ---------- routes ----------
@app.get("/")
def root():
    return {"status": "api is running"}

@app.get("/health")
def health():
    return {"ok": True}


