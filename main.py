import os
import sys
import time
import uuid
from typing import Dict, List

from PIL import Image
from tools.painter import mask_painter
import cv2
import gdown
import numpy as np
import psutil
import requests
import torch
import torchvision
from fastapi import FastAPI, UploadFile, File, HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response,HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# -----------------------------
# 路径 & 模块初始化
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原来是 ..，现在改成当前目录下的 frontend
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

RESULT_DIR = os.path.join(BASE_DIR, "result")
TRACK_DIR = os.path.join(RESULT_DIR, "track")
INPAINT_DIR = os.path.join(RESULT_DIR, "inpaint")

os.makedirs(TRACK_DIR, exist_ok=True)
os.makedirs(INPAINT_DIR, exist_ok=True)
sys.path.append(os.path.join(BASE_DIR, "tracker"))
sys.path.append(os.path.join(BASE_DIR, "tracker", "model"))

try:
    # 只是为了触发 mmcv 安装，不一定直接用到
    from mmcv.cnn import ConvModule  # noqa: F401
except Exception:
    os.system("mim install mmcv")

from track_anything import TrackingAnything, parse_augment
from tools.painter import mask_painter

RESULT_DIR = os.path.join(BASE_DIR, "result")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# FastAPI 初始化
# -----------------------------
app = FastAPI(title="Track Anything API", version="1.0")

# CORS：方便你前端单独起在别的端口（如 5500/5173 等）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 实际线上请收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件：用于访问 result/ 下的输出视频
#app.mount("/static", StaticFiles(directory=RESULT_DIR), name="static")
app.mount("/static/track", StaticFiles(directory=TRACK_DIR), name="static_track")

app.mount("/static/inpaint", StaticFiles(directory=INPAINT_DIR), name="static_inpaint")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """浏览器访问 http://127.0.0.1:8000 时返回前端页面"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(content="index.html 不存在，请检查 FRONTEND_DIR 路径", status_code=500)
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# 权重下载工具
# -----------------------------
def download_checkpoint(url: str, folder: str, filename: str) -> str:
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        print(f"Downloading checkpoint {filename} ...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download finished:", filepath)
    return filepath


def download_checkpoint_from_google_drive(file_id: str, folder: str, filename: str) -> str:
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        print(
            "Downloading checkpoints from Google Drive..."
            "\n如果进度条不动，可以手动下载："
            "https://github.com/MCG-NKU/E2FGVI (E2FGVI-HQ 模型)"
        )
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Download finished:", filepath)
    return filepath


# -----------------------------
# args & 模型初始化（替代 Gradio 部分）
# -----------------------------
# 防止 parse_augment 读到 uvicorn 的命令行参数
orig_argv = sys.argv.copy()
sys.argv = [orig_argv[0]]
args = parse_augment()
sys.argv = orig_argv

if not hasattr(args, "sam_model_type"):
    args.sam_model_type = "vit_h"
if not hasattr(args, "mask_save"):
    args.mask_save = False

args.port = 12212
args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

SAM_checkpoint_dict = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
SAM_checkpoint_url_dict = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

sam_model_type = getattr(args, "sam_model_type", "vit_h")
if sam_model_type not in SAM_checkpoint_dict:
    sam_model_type = "vit_h"

sam_checkpoint_name = SAM_checkpoint_dict[sam_model_type]
sam_checkpoint_url = SAM_checkpoint_url_dict[sam_model_type]
xmem_checkpoint_name = "XMem-s012.pth"
xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
e2fgvi_checkpoint_name = "E2FGVI-HQ-CVPR22.pth"
e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"

SAM_checkpoint = download_checkpoint(sam_checkpoint_url, CHECKPOINT_DIR, sam_checkpoint_name)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, CHECKPOINT_DIR, xmem_checkpoint_name)
e2fgvi_checkpoint = download_checkpoint_from_google_drive(
    e2fgvi_checkpoint_id, CHECKPOINT_DIR, e2fgvi_checkpoint_name
)

print("Initializing TrackingAnything model ...")
model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args)
print("Model ready.")


# -----------------------------
# Session 状态结构
# -----------------------------
class SessionState:
    def __init__(
        self,
        session_id: str,
        video_path: str,
        video_name: str,
        frames: List[np.ndarray],
        fps: float,
        mask_save: bool,
    ) -> None:
        self.session_id = session_id
        self.video_path = video_path
        self.video_state = {
            "user_name": time.time(),
            "video_name": video_name,
            "origin_images": frames,
            "painted_images": frames.copy(),
            "masks": [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8)] * len(frames),
            "logits": [None] * len(frames),
            "select_frame_number": 0,
            "fps": fps,
        }
        self.interactive_state = {
            "inference_times": 0,
            "negative_click_times": 0,
            "positive_click_times": 0,
            "mask_save": mask_save,
            "multi_mask": {
                "mask_names": [],
                "masks": [],
            },
            "track_end_number": None,
            "resize_ratio": 1.0,
        }
        # click_state = [[x,y...],[label...]]
        self.click_state = [[], []]


SESSIONS: Dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# -----------------------------
# 工具函数：抽帧 & 写视频 & 点集
# -----------------------------
def extract_frames(video_path: str):
    frames: List[np.ndarray] = []
    fps = 30.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_val = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_val) if fps_val and fps_val > 0 else 30.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_memory_usage = psutil.virtual_memory().percent
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if current_memory_usage > 90:
            cap.release()
            raise MemoryError("Memory usage is too high (>90%). Stopped video extraction early.")
    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted from video.")
    return frames, fps


def generate_video_from_frames(frames, output_path, fps=30):
    import numpy as np
    import torch
    import torchvision
    import os

    frames_np = np.asarray(frames, dtype=np.uint8)

    if frames_np.ndim != 4 or frames_np.shape[-1] != 3:
        raise ValueError(f"frames 形状不对，期望 (T,H,W,3)，实际 {frames_np.shape}")

    frames_t = torch.from_numpy(frames_np)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("[VIDEO] writing:", output_path)
    torchvision.io.write_video(output_path, frames_t, fps=fps, video_codec="libx264")
    print("[VIDEO] done, exists:", os.path.exists(output_path))

    return output_path



def add_click_to_state(click_state, x: int, y: int, label: int):
    # click_state = [[(x,y)...], [label...]]
    click_state[0].append([x, y])
    click_state[1].append(label)
    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt


# -----------------------------
# Pydantic 请求/响应模型
# -----------------------------
class UploadVideoResponse(BaseModel):
    session_id: str
    video_info: str
    total_frames: int
    fps: float
    height: int
    width: int


class SelectFrameRequest(BaseModel):
    session_id: str
    frame_index: int  # 0-based


class TrackEndRequest(BaseModel):
    session_id: str
    frame_index: int  # 0-based, 作为切片右边界


class ResizeRatioRequest(BaseModel):
    session_id: str
    resize_ratio: float  # 0.02 ~ 1.0


class SamClickRequest(BaseModel):
    session_id: str
    x: int
    y: int
    label: int  # 1=positive, 0=negative


class SessionOnlyRequest(BaseModel):
    session_id: str


class MaskNamesRequest(BaseModel):
    session_id: str
    mask_names: List[str] = []  # ["mask_001", "mask_002", ...]


class TrackRequest(BaseModel):
    session_id: str
    mask_names: List[str] = []


class InpaintRequest(BaseModel):
    session_id: str
    mask_names: List[str] = []


# -----------------------------
# API：上传视频并抽帧
# -----------------------------
@app.post("/api/upload_video", response_model=UploadVideoResponse)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(file.filename)[1] or ".mp4"
    session_id = str(uuid.uuid4())
    saved_name = f"{session_id}{ext}"
    video_path = os.path.join(UPLOAD_DIR, saved_name)

    # 保存上传文件
    with open(video_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    # 抽帧
    try:
        frames, fps = extract_frames(video_path)
    except MemoryError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 初始化 session
    original_name = os.path.basename(file.filename)
    session = SessionState(
        session_id=session_id,
        video_path=video_path,
        video_name=original_name,
        frames=frames,
        fps=fps,
        mask_save=args.mask_save,
    )
    SESSIONS[session_id] = session

    # 初始化 SAM 图像
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(frames[0])

    h, w = frames[0].shape[:2]
    video_info = f"Video Name: {original_name}, FPS: {fps:.2f}, Total Frames: {len(frames)}, Image Size: ({h}, {w})"

    return UploadVideoResponse(
        session_id=session_id,
        video_info=video_info,
        total_frames=len(frames),
        fps=float(fps),
        height=int(h),
        width=int(w),
    )


# -----------------------------
# API：返回一帧图像（用于前端 <img> 显示）
# -----------------------------
@app.get("/api/frame/{session_id}/{frame_index}")
def get_frame(
    session_id: str,
    frame_index: int,
    img_type: str = Query("painted", alias="type"),  # 前端还是用 ?type=painted
):
    session = get_session(session_id)
    vs = session.video_state

    # 1. 基本检查
    if vs["origin_images"] is None or vs["painted_images"] is None:
        raise HTTPException(status_code=500, detail="video_state 里还没有帧，请先上传并解析视频")

    total = len(vs["origin_images"])
    if frame_index < 0 or frame_index >= total:
        raise HTTPException(
            status_code=400,
            detail=f"frame_index 必须在 [0, {total-1}]，当前是 {frame_index}",
        )

    # 2. 取出对应帧
    if img_type == "origin":
        frame = vs["origin_images"][frame_index]
    else:
        frame = vs["painted_images"][frame_index]

    # 3. debug 打印（注意这里不用内置 type()，直接用 __class__）
    print(
        "[get_frame] img_type=",
        img_type,
        "index=",
        frame_index,
        "frame class:",
        frame.__class__,
    )

    # 4. 防守式检查
    if frame is None:
        raise HTTPException(
            status_code=500,
            detail="当前帧是 None，说明视频读取失败或还未初始化完成",
        )

    import numpy as np

    if not isinstance(frame, np.ndarray):
        raise HTTPException(
            status_code=500,
            detail=f"当前帧不是 numpy.ndarray，而是 {type(frame)}，请检查视频读取/处理流程",
        )

    # 5. 正常转换
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="OpenCV 编码 JPEG 失败")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

# -----------------------------
# API：选择起始帧
# -----------------------------
@app.post("/api/select_frame")
def select_frame(req: SelectFrameRequest):
    session = get_session(req.session_id)
    vs = session.video_state
    total = len(vs["origin_images"])
    idx = int(req.frame_index)
    if idx < 0 or idx >= total:
        raise HTTPException(status_code=400, detail=f"frame_index must be in [0, {total-1}]")

    vs["select_frame_number"] = idx
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(vs["origin_images"][idx])
    return {"ok": True, "select_frame_number": idx}


# -----------------------------
# API：设置 tracking 结束帧
# -----------------------------
@app.post("/api/set_track_end")
def set_track_end(req: TrackEndRequest):
    session = get_session(req.session_id)
    vs = session.video_state
    total = len(vs["origin_images"])
    idx = int(req.frame_index)
    if idx < 0 or idx >= total:
        raise HTTPException(status_code=400, detail=f"frame_index must be in [0, {total-1}]")
    session.interactive_state["track_end_number"] = idx
    return {"ok": True, "track_end_number": idx}


# -----------------------------
# API：设置 inpaint resize ratio
# -----------------------------
@app.post("/api/set_resize_ratio")
def set_resize_ratio(req: ResizeRatioRequest):
    session = get_session(req.session_id)
    ratio = float(req.resize_ratio)
    ratio = max(0.02, min(ratio, 1.0))
    session.interactive_state["resize_ratio"] = ratio
    return {"ok": True, "resize_ratio": ratio}


# -----------------------------
# API：点击 SAM（正负样本）
# -----------------------------
@app.post("/api/sam_click")
def sam_click(req: SamClickRequest):
    session = get_session(req.session_id)
    vs = session.video_state
    istate = session.interactive_state
    click_state = session.click_state

    frame_idx = vs["select_frame_number"]
    if req.label == 1:
        istate["positive_click_times"] += 1
    else:
        istate["negative_click_times"] += 1

    # 重新设置 SAM 的图像
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(vs["origin_images"][frame_idx])
    prompt = add_click_to_state(click_state, req.x, req.y, req.label)

    mask, logit, painted_image = model.first_frame_click(
        image=vs["origin_images"][frame_idx],
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )

    # ⭐⭐ 关键：把 painted_image 转成 numpy
    if isinstance(painted_image, Image.Image):
        painted_np = np.array(painted_image)
    elif isinstance(painted_image, np.ndarray):
        painted_np = painted_image
    else:
        painted_np = np.array(painted_image)

    if painted_np.dtype != np.uint8:
        painted_np = painted_np.astype(np.uint8)

    vs["masks"][frame_idx] = mask
    vs["logits"][frame_idx] = logit
    vs["painted_images"][frame_idx] = painted_np  # 只存 numpy

    return {"ok": True}



# -----------------------------
# API：添加当前帧 mask 到 multi_mask
# -----------------------------
@app.post("/api/add_mask")
def add_mask(req: SessionOnlyRequest):
    session = get_session(req.session_id)
    vs = session.video_state
    istate = session.interactive_state
    frame_idx = vs["select_frame_number"]

    mask = vs["masks"][frame_idx]
    if mask is None:
        raise HTTPException(status_code=400, detail="No mask on current frame. Click image first.")

    istate["multi_mask"]["masks"].append(mask)
    name = f"mask_{len(istate['multi_mask']['masks']):03d}"
    istate["multi_mask"]["mask_names"].append(name)

    return {"ok": True, "mask_names": istate["multi_mask"]["mask_names"]}


# -----------------------------
# API：清空点击
# -----------------------------
@app.post("/api/clear_clicks")
def clear_clicks(req: SessionOnlyRequest):
    session = get_session(req.session_id)
    session.click_state = [[], []]
    vs = session.video_state
    frame_idx = vs["select_frame_number"]
    vs["painted_images"][frame_idx] = vs["origin_images"][frame_idx]
    return {"ok": True}


# -----------------------------
# API：清空所有 multi_mask
# -----------------------------
@app.post("/api/remove_all_masks")
def remove_all_masks(req: SessionOnlyRequest):
    session = get_session(req.session_id)
    session.interactive_state["multi_mask"]["mask_names"] = []
    session.interactive_state["multi_mask"]["masks"] = []
    return {"ok": True, "mask_names": []}


# -----------------------------
# API：根据选中 mask 显示叠加效果
# -----------------------------
@app.post("/api/show_mask")
def show_mask(req: MaskNamesRequest):
    session = get_session(req.session_id)
    vs = session.video_state
    istate = session.interactive_state
    frame_idx = vs["select_frame_number"]

    select_frame = vs["origin_images"][frame_idx].copy()

    mask_names = sorted(req.mask_names)
    for name in mask_names:
        try:
            mask_number = int(name.split("_")[1]) - 1
            mask = istate["multi_mask"]["masks"][mask_number]
            select_frame = mask_painter(select_frame, mask.astype("uint8"), mask_color=mask_number + 2)
        except Exception:
            continue

    vs["painted_images"][frame_idx] = select_frame
    return {"ok": True}


# -----------------------------
# API：Tracking 视频
# -----------------------------
@app.post("/api/track")
def vos_tracking(req: TrackRequest):
    session = get_session(req.session_id)
    vs = session.video_state
    istate = session.interactive_state

    model.xmem.clear_memory()

    start = vs["select_frame_number"]
    total = len(vs["origin_images"])
    end = istate["track_end_number"] if istate["track_end_number"] is not None else total
    end = max(start + 1, min(end, total))  # 至少跟踪一帧

    following_frames = vs["origin_images"][start:end]
    if len(following_frames) == 0:
        raise HTTPException(status_code=400, detail="No frames to track.")

    mask_dropdown = req.mask_names or []

    # 1. 构造 template_mask（起始帧的多掩码合成）
    if istate["multi_mask"]["masks"]:
        if not mask_dropdown:
            mask_dropdown = ["mask_001"]
        mask_dropdown = sorted(mask_dropdown)
        first_idx = int(mask_dropdown[0].split("_")[1]) - 1
        template_mask = istate["multi_mask"]["masks"][first_idx] * (first_idx + 1)
        for name in mask_dropdown[1:]:
            mask_number = int(name.split("_")[1]) - 1
            template_mask = np.clip(
                template_mask + istate["multi_mask"]["masks"][mask_number] * (mask_number + 1),
                0,
                mask_number + 1,
            )
        vs["masks"][start] = template_mask
    else:
        template_mask = vs["masks"][start]

    # 2. 防止 template_mask 全 0
    if len(np.unique(template_mask)) == 1:
        template_mask[0][0] = 1

    fps = vs["fps"]

    # 3. XMem 做跟踪，得到后面所有帧的 mask
    masks, logits, painted_images = model.generator(
        images=following_frames,
        template_mask=template_mask,
    )
    model.xmem.clear_memory()

    # 4. 把跟踪结果写回 video_state（★ 你之前缺了这步）
    masks = list(masks)
    logits = list(logits)

    vs["masks"][start:end] = masks
    vs["logits"][start:end] = logits

    istate["inference_times"] += 1

    # 5. 重新根据 mask 给整段视频上色
    painted_all = []
    for img, mask in zip(vs["origin_images"], vs["masks"]):
        # mask 可能是 None 或全 0，这两种情况都直接用原图
        if mask is None or np.max(mask) == 0:
            painted_all.append(img)
        else:
            painted = mask_painter(img, mask.astype("uint8"))
            painted_all.append(painted)

    vs["painted_images"] = painted_all

    # 6. 保存视频到 TRACK_DIR
    video_name = vs.get("video_name", "output.mp4")
    out_name = f"{req.session_id}_{video_name}"
    out_path = os.path.join(TRACK_DIR, out_name)

    print("[TRACK] write to:", out_path)
    generate_video_from_frames(vs["painted_images"], out_path, fps=int(round(fps)))
    print("[TRACK] exists:", os.path.exists(out_path))

    video_url = f"/static/track/{out_name}"
    return {"ok": True, "video_url": video_url}

# -----------------------------
# API：Inpaint 视频
# -----------------------------
@app.post("/api/inpaint")
def inpaint(req: InpaintRequest):
    session = get_session(req.session_id)
    vs = session.video_state
    istate = session.interactive_state

    # 1. 取出整段视频帧 & mask
    frames = np.asarray(vs["origin_images"], dtype=np.uint8)   # (T, H, W, 3)
    fps = vs["fps"]
    inpaint_masks = np.asarray(vs["masks"], dtype=np.uint8)    # (T, H, W) 或 (T, H, W, 1)

    # 如果是 (T, H, W, 1)，压掉最后一维
    if inpaint_masks.ndim == 4 and inpaint_masks.shape[-1] == 1:
        inpaint_masks = inpaint_masks[..., 0]

    # 2. 处理前端勾选的掩码编号（mask_001, mask_002...）
    mask_dropdown = req.mask_names or []
    if not mask_dropdown:
        mask_dropdown = ["mask_001"]   # 默认用第一个掩码

    mask_dropdown.sort()
    selected_numbers = [int(name.split("_")[1]) for name in mask_dropdown]

    # 3. 只保留选中的 mask id，其它全部置 0
    max_id = int(inpaint_masks.max())
    for i in range(1, max_id + 1):
        if i not in selected_numbers:
            inpaint_masks[inpaint_masks == i] = 0

    # 4. 调用 E2FGVI 做整段 inpaint
    try:
        inpainted_frames = model.baseinpainter.inpaint(
            frames,
            inpaint_masks,
            ratio=istate["resize_ratio"],
        )  # numpy, (T, H, W, 3)
    except Exception as e:
        print("[INPAINT] error:", e)
        inpainted_frames = frames  # 出错就用原视频兜底

    # 5. 同步回 video_state，方便后续继续操作
    #    转回 list[np.ndarray] 的形式
    vs["origin_images"] = [f for f in inpainted_frames]
    vs["painted_images"] = [f for f in inpainted_frames]

    # 6. 保存视频到全局 INPAINT_DIR
    video_name = vs.get("video_name", "output.mp4")
    out_name = f"{req.session_id}_{video_name}"
    out_path = os.path.join(INPAINT_DIR, out_name)

    print("[INPAINT] write to:", out_path)
    generate_video_from_frames(inpainted_frames, out_path, fps=int(round(fps)))
    print("[INPAINT] exists:", os.path.exists(out_path))

    video_url = f"/static/inpaint/{out_name}"
    return {"ok": True, "video_url": video_url}



# -----------------------------
# 本地运行入口（可选）
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
