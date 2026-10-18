"""
plugins/ocr_layout.py — PicoDet-S_layout_17cls 版面分析（TensorRT）。

OCR 插件的可选增强：layout 引擎把页面切成 17 类版面区域（标题、正文、
表格、页眉页脚……），OCR item 按中心点归属到区域后，payload 附带
regions / summary 供下游做结构化阅读。

模型约定（layout_nonms_fp32.engine，静态形状）：

    输入  image        (1,3,480,480) float32 — BGR 直接 resize 到 480x480
                       （不保持宽高比），/255，再减 ImageNet mean 除 std，
                       transpose 成 NCHW
    输入  scale_factor (1,2)         float32 — 固定 [[1.0, 1.0]]
    输出  scores       (1,17,N)      float32 — 每类置信度（NMS 前）
    输出  boxes        (1,N,4)       float32 — xyxy，480x480 输入坐标系，
                       映射回原图 x*W/480, y*H/480
    后处理            numpy 版 MultiClassNMS：score 阈值（默认 0.3）、
                      IoU 阈值 0.5、按类独立 NMS、keep_top_k=100

引擎有两个输入，共享的 utils.tensorrt_runtime.TensorRTEngine 只封装
单输入引擎，所以这里基于 CudaRuntime + read_engine_file 驱动反序列化，
流程与 TensorRTEngine 保持一致（同一把锁保护 execute + 显存释放）。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from utils.tensorrt_runtime import (
    CudaRuntime,
    TensorRTError,
    read_engine_file,
    trt_dtype_to_numpy,
)

_log = logging.getLogger(__name__)

LAYOUT_INPUT_SIZE = 480
LAYOUT_NUM_CLASSES = 17
LAYOUT_MEAN = (0.485, 0.456, 0.406)
LAYOUT_STD = (0.229, 0.224, 0.225)
DEFAULT_LAYOUT_SCORE_THRESH = 0.3
LAYOUT_NMS_IOU_THRESH = 0.5
LAYOUT_KEEP_TOP_K = 100

# 引擎输出的类顺序（paddle PicoDet-S_layout_17cls）。
LAYOUT_LABELS = (
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "table_title",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
)

# 角色中文映射（build_layout_summary 用）。
LAYOUT_ROLE_ZH = {
    "doc_title": "文档标题",
    "paragraph_title": "段落标题",
    "text": "正文",
    "content": "正文",
    "abstract": "正文",
    "number": "数据指标",
    "table": "表格",
    "table_title": "表格标题",
    "figure_title": "图注",
    "header": "页眉",
    "footer": "页脚",
    "footnote": "脚注",
    "formula": "公式",
    "algorithm": "算法",
    "reference": "参考文献",
    "image": "图片区域",
    "seal": "印章",
}

# 摘要摘取规则：标题类全取；正文类取前 25 字 + 省略号；其余（number 等短
# 文本）全取。
_BODY_LABELS = frozenset({"text", "content", "abstract"})
_BODY_EXCERPT_LIMIT = 25


class LayoutTensorRTSession:
    """PicoDet-S_layout_17cls 的专用 TensorRT session：BGR 图进，版面区域出。

    引擎为静态形状、两个输入（image + scale_factor），device buffer 在建
    构时一次分配。`infer()` 返回按 score 降序的区域列表，bbox 为原图像素
    坐标：

        [{"label": "text", "score": 0.93,
          "bbox": [x1, y1, x2, y2]}, ...]
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device_id: int = 0,
        score_thresh: float = DEFAULT_LAYOUT_SCORE_THRESH,
    ):
        import numpy as np
        import tensorrt as trt

        self._np = np
        self._trt = trt
        self._lock = threading.Lock()
        self._score_thresh = float(score_thresh)
        self._path = str(engine_path)
        self._cuda: CudaRuntime | None = None
        self._runtime = None
        self._engine = None
        self._context = None
        self._device_buffers: dict[str, int] = {}
        self._inputs: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        self._outputs: dict[str, np.ndarray] = {}

        if not Path(engine_path).is_file():
            raise FileNotFoundError(
                f"layout TensorRT engine file missing: {engine_path}"
            )
        mean = np.asarray(LAYOUT_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(LAYOUT_STD, dtype=np.float32).reshape(1, 1, 3)
        self._mean = mean
        self._std_inv = 1.0 / std

        _metadata, serialized = read_engine_file(engine_path)
        try:
            self._cuda = CudaRuntime(device_id)
            self._logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(serialized)
            del serialized
            if self._engine is None:
                raise TensorRTError(
                    f"failed to deserialize TensorRT engine: {self._path}"
                )
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise TensorRTError(
                    f"failed to create TensorRT context: {self._path}"
                )

            names = [
                self._engine.get_tensor_name(index)
                for index in range(self._engine.num_io_tensors)
            ]
            input_names = [
                name for name in names
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            ]
            output_names = [name for name in names if name not in input_names]
            if len(input_names) != 2 or not output_names:
                raise TensorRTError(
                    "layout TensorRT engine must have exactly 2 inputs and at "
                    f"least 1 output; got {names} ({self._path})"
                )

            # 静态形状引擎：解析形状并一次性分配 device buffer。
            for name in input_names:
                shape = tuple(
                    int(v) for v in self._engine.get_tensor_shape(name)
                )
                if any(v <= 0 for v in shape):
                    raise TensorRTError(
                        f"layout engine input {name} has non-static shape "
                        f"{shape} ({self._path})"
                    )
                dtype = trt_dtype_to_numpy(
                    trt, self._engine.get_tensor_dtype(name)
                )
                self._inputs[name] = (shape, dtype)
                if not self._context.set_input_shape(name, shape):
                    raise TensorRTError(
                        f"TensorRT rejected input shape {shape} for {name} "
                        f"({self._path})"
                    )
            for name in output_names:
                shape = tuple(
                    int(v) for v in self._context.get_tensor_shape(name)
                )
                if any(v <= 0 for v in shape):
                    raise TensorRTError(
                        f"layout engine output {name} has non-static shape "
                        f"{shape} ({self._path})"
                    )
                dtype = trt_dtype_to_numpy(
                    trt, self._engine.get_tensor_dtype(name)
                )
                self._outputs[name] = np.empty(shape, dtype=dtype)

            for name, (shape, dtype) in self._inputs.items():
                array = np.empty(shape, dtype=dtype)
                pointer = self._cuda.malloc(array.nbytes)
                self._device_buffers[name] = pointer
                self._context.set_tensor_address(name, pointer)
            for name, array in self._outputs.items():
                pointer = self._cuda.malloc(array.nbytes)
                self._device_buffers[name] = pointer
                self._context.set_tensor_address(name, pointer)
        except Exception:
            self.close()
            raise

    def _run(self, image_array, scale_factor_array) -> list[np.ndarray]:
        np = self._np
        with self._lock:
            if self._context is None or self._cuda is None:
                raise TensorRTError(f"TensorRT engine is closed ({self._path})")
            cuda = self._cuda
            feeds = []
            for name, (shape, dtype) in self._inputs.items():
                if len(shape) == 4:
                    array = np.ascontiguousarray(image_array, dtype=dtype)
                else:
                    array = np.ascontiguousarray(scale_factor_array, dtype=dtype)
                if array.shape != shape:
                    raise ValueError(
                        f"layout engine input {name} shape mismatch: "
                        f"{array.shape} vs {shape}"
                    )
                feeds.append(array)

            for name, array in zip(self._inputs, feeds):
                cuda.copy_host_to_device(self._device_buffers[name], array)
            if not self._context.execute_async_v3(cuda.stream_handle):
                raise TensorRTError(f"TensorRT execute_async_v3 failed ({self._path})")
            outputs = []
            for name in self._outputs:
                output = self._outputs[name]
                cuda.copy_device_to_host(output, self._device_buffers[name])
                outputs.append(output.copy())
            cuda.synchronize()
            return outputs

    def _postprocess(self, scores, boxes, width: int, height: int) -> list[dict]:
        np = self._np
        scores = scores[0]
        boxes = boxes[0].astype(np.float32, copy=False)
        scale = np.asarray(
            [
                width / LAYOUT_INPUT_SIZE,
                height / LAYOUT_INPUT_SIZE,
                width / LAYOUT_INPUT_SIZE,
                height / LAYOUT_INPUT_SIZE,
            ],
            dtype=np.float32,
        )
        boxes = boxes * scale
        return self._multiclass_nms(scores, boxes)

    def _multiclass_nms(self, scores, boxes) -> list[dict]:
        """对齐 paddle MultiClassNMS：按类独立 NMS，全局 keep_top_k 截断。"""
        np = self._np
        kept: list[tuple[int, int, float]] = []
        for cls in range(min(scores.shape[0], LAYOUT_NUM_CLASSES)):
            cls_scores = scores[cls]
            indices = np.nonzero(cls_scores >= self._score_thresh)[0]
            if indices.size == 0:
                continue
            indices = indices[np.argsort(-cls_scores[indices], kind="stable")]
            while indices.size:
                top = int(indices[0])
                kept.append((cls, top, float(cls_scores[top])))
                rest = indices[1:]
                if rest.size:
                    ious = _boxes_iou(boxes[top], boxes[rest])
                    indices = rest[ious <= LAYOUT_NMS_IOU_THRESH]
                else:
                    break
        kept.sort(key=lambda entry: -entry[2])
        kept = kept[:LAYOUT_KEEP_TOP_K]
        detections = [
            {
                "label": LAYOUT_LABELS[cls] if cls < len(LAYOUT_LABELS) else str(cls),
                "score": score,
                "bbox": [float(v) for v in boxes[index]],
            }
            for cls, index, score in kept
        ]
        return _merge_overlapping_regions(detections)

    def infer(self, image) -> list[dict]:
        """BGR uint8 原图进，返回 NMS 后的版面区域（原图坐标，score 降序）。"""
        import cv2

        np = self._np
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return []
        # keep_ratio=false：直接拉到 480x480，与训练时 resize 一致。
        resized = cv2.resize(
            image,
            (LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        array = resized.astype(np.float32) * (1.0 / 255.0)
        array = (array - self._mean) * self._std_inv
        nchw = np.ascontiguousarray(array.transpose(2, 0, 1)[None])
        scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)
        outputs = self._run(nchw, scale_factor)
        scores, boxes = None, None
        for output in outputs:
            if output.ndim == 3 and output.shape[1] == LAYOUT_NUM_CLASSES:
                scores = output
            elif output.ndim == 3 and output.shape[2] == 4:
                boxes = output
        if scores is None or boxes is None:
            raise TensorRTError(
                f"layout engine outputs not recognized: "
                f"{[tuple(o.shape) for o in outputs]} ({self._path})"
            )
        return self._postprocess(scores, boxes, width, height)

    def close(self) -> None:
        # 与 TensorRTEngine.close 相同：在 infer 的同一把锁下释放显存，
        # 避免 close 撞上正在执行的 inference。
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            cuda = getattr(self, "_cuda", None)
            if cuda is not None:
                for name, pointer in list(self._device_buffers.items()):
                    try:
                        cuda.free(pointer)
                    except Exception:
                        _log.debug("cudaFree failed for %s", name, exc_info=True)
            self._device_buffers = {}
            self._inputs = {}
            self._outputs = {}
            self._context = None
            self._engine = None
            self._runtime = None
            if cuda is not None:
                self._cuda = None
                try:
                    cuda.close()
                except Exception:
                    _log.debug("cudaStreamDestroy failed", exc_info=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can unload CUDA before Python finalizers run.
            pass


def _boxes_iou(box, others):
    """单个 xyxy box 对一批 xyxy box 的 IoU（numpy）。"""
    import numpy as np

    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = (
        np.maximum(0.0, others[:, 2] - others[:, 0])
        * np.maximum(0.0, others[:, 3] - others[:, 1])
    )
    union = area_a + area_b - inter
    return np.divide(
        inter, union, out=np.zeros_like(inter), where=union > 0
    )


def _item_region_iou(item_bbox, region_bbox) -> float:
    """单个 OCR item bbox 与 layout region bbox 的 IoU（纯 Python）。"""
    ax1, ay1, ax2, ay2 = item_bbox
    bx1, by1, bx2, by2 = region_bbox
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_overlapping_regions(detections: list[dict]) -> list[dict]:
    """把 NMS 后的重叠检测合并成互不重叠的版面区域。

    PicoDet 按类独立 NMS，同一视觉区域常以多个类别（doc_title 与
    paragraph_title）或多个粒度（整栏一个大框 + 每行一个小框）重复出现，
    直接归属会让 item 重复计数、总结里的区域数虚高。这里按 score 降序
    贪心处理，与已接受区域重叠时（IoU > 0.5，或小块被大块包含过半）：
    同类合并（bbox 取并集），异类丢弃（保留高分者）；否则接受为新区域。
    """
    accepted: list[dict] = []
    for det in sorted(detections, key=lambda d: -d["score"]):
        merged = False
        for region in accepted:
            iou = _item_region_iou(det["bbox"], region["bbox"])
            containment = _containment_ratio(det["bbox"], region["bbox"])
            if iou <= 0.5 and containment <= 0.5:
                continue
            if det["label"] == region["label"]:
                dx1, dy1, dx2, dy2 = det["bbox"]
                rx1, ry1, rx2, ry2 = region["bbox"]
                region["bbox"] = [
                    min(rx1, dx1), min(ry1, dy1),
                    max(rx2, dx2), max(ry2, dy2),
                ]
                region["score"] = max(region["score"], det["score"])
            merged = True
            break
        if not merged:
            accepted.append({"label": det["label"], "score": det["score"],
                             "bbox": list(det["bbox"])})
    return accepted


def _containment_ratio(inner, outer) -> float:
    """inner bbox 被 outer bbox 覆盖的面积占 inner 自身的比例。"""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    cx1, cy1 = max(ix1, ox1), max(iy1, oy1)
    cx2, cy2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / area if area > 0 else 0.0


# 包含 OCR 文字的区域不该按图片/印章角色描述（场景图常被整体误判为 image）。
_NON_TEXTUAL_LABELS = frozenset({"image", "seal"})


def relabel_textual_regions(
    regions: list[dict], assigned: dict[int, list[int]]
) -> None:
    """label 为 image/seal 却装下了 OCR 文字的区域，改标为 text（原地修改）。"""
    for region_index, region in enumerate(regions):
        if (
            region.get("label") in _NON_TEXTUAL_LABELS
            and assigned.get(region_index)
        ):
            region["label"] = "text"


def rule_based_regions(items: list[dict], image_size) -> list[dict]:
    """从 OCR items 推导版面区域：字号定标题、x 空白分栏、底部定页脚。

    layout 模型（文档域训练）在自然场景图上常退化成单个大区域，此时用
    OCR 结果自己做结构分析。区域格式与 layout 引擎输出一致，可直接进
    assign_regions / build_layout_summary。
    """
    if not items:
        return []
    width = float(image_size[0]) if image_size else 0.0
    height = float(image_size[1]) if image_size else 0.0
    heights = sorted(
        max(1, it["bbox"][3] - it["bbox"][1]) for it in items
    )
    median_h = heights[len(heights) // 2]

    def union(its, label):
        return {
            "label": label,
            "score": 1.0,
            "bbox": [
                min(it["bbox"][0] for it in its),
                min(it["bbox"][1] for it in its),
                max(it["bbox"][2] for it in its),
                max(it["bbox"][3] for it in its),
            ],
        }

    regions = []
    used = set()
    # 标题：字号 ≥ 1.4 倍中位、字数 ≥ 2、在页面上半部、绝对高度 ≥ 2% 页高
    # （限制条件防把小票/地图里偏大的行误判为标题）
    titles = [
        it for it in items
        if (it["bbox"][3] - it["bbox"][1] >= 1.4 * median_h
            and it["bbox"][3] - it["bbox"][1] >= 0.02 * (height or 10**9)
            and (it["bbox"][1] + it["bbox"][3]) / 2.0 < 0.5 * (height or 10**9)
            and len(str(it.get("text") or "").strip()) >= 2)
    ]
    for it in titles:
        regions.append(union([it], "paragraph_title"))
        used.add(id(it))
    # 页脚：底部 8% 内、字号不超过 0.9 倍中位
    if height:
        for it in items:
            if id(it) in used:
                continue
            if it["bbox"][1] >= height * 0.92 and \
                    it["bbox"][3] - it["bbox"][1] <= 0.9 * median_h:
                regions.append(union([it], "footer"))
                used.add(id(it))
    # 正文：按最大垂直空白间隙分栏（间隙 > 8% 页宽且两侧各 ≥2 行才切）
    body = [it for it in items if id(it) not in used]
    if body:
        columns = _split_columns(body, width)
        for column in columns:
            regions.append(union(column, "text"))
    return regions


def _split_columns(items, page_width):
    """按 x 间隙把 item 序列切成栏（单层级，最多切一刀）。"""
    if len(items) < 4 or page_width <= 0:
        return [items]
    ordered = sorted(items, key=lambda it: it["bbox"][0])
    best_gap, best_at = 0.0, -1
    for i in range(len(ordered) - 1):
        gap = ordered[i + 1]["bbox"][0] - ordered[i]["bbox"][2]
        if gap > best_gap:
            best_gap, best_at = gap, i
    if best_at < 0 or best_gap <= page_width * 0.08:
        return [items]
    left, right = ordered[: best_at + 1], ordered[best_at + 1:]
    if len(left) >= 2 and len(right) >= 2:
        return [left, right]
    return [items]


def assign_regions(
    items: list[dict], regions: list[dict]
) -> dict[int, list[int]]:
    """把 OCR item 归属到 layout region，返回 {region_index: [item_indices]}。

    归属规则：item bbox 中心点落在 region bbox 内 → 归该 region（多个包含
    时取 IoU 最大者，即最小的包含区域）；中心点不在任何 region 内 → 归 IoU
    最大的 region，但 IoU 必须大于 0。一个 region 可容纳多个 item，未被
    任何 region 覆盖的 item 不出现在返回值里。
    """
    assigned: dict[int, list[int]] = {}
    for index, item in enumerate(items):
        bbox = item.get("bbox") or (0, 0, 0, 0)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        containing: list[tuple[float, int]] = []
        fallback_index = -1
        fallback_iou = 0.0
        for region_index, region in enumerate(regions):
            region_bbox = region.get("bbox") or (0, 0, 0, 0)
            iou = _item_region_iou(bbox, region_bbox)
            rx1, ry1, rx2, ry2 = region_bbox
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                containing.append((iou, region_index))
            if iou > fallback_iou:
                fallback_iou = iou
                fallback_index = region_index
        if containing:
            region_index = max(containing)[1]
        elif fallback_iou > 0.0:
            region_index = fallback_index
        else:
            continue
        assigned.setdefault(region_index, []).append(index)
    return assigned


def _region_position(bbox, image_size) -> str:
    """region bbox 中心相对页面九宫格的方位。"""
    x1, y1, x2, y2 = bbox
    width, height = image_size
    if width <= 0 or height <= 0:
        return ""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    horizontal = "左" if cx < width / 3 else ("右" if cx > 2 * width / 3 else "")
    vertical = "上" if cy < height / 3 else ("下" if cy > 2 * height / 3 else "")
    if vertical and horizontal:
        return f"{horizontal}{vertical}"
    if vertical:
        return "顶部" if vertical == "上" else "底部"
    if horizontal:
        return f"{horizontal}侧"
    return "居中"


def _region_role_zh(label: str) -> str:
    return LAYOUT_ROLE_ZH.get(label, "内容区域")


def _region_excerpt(label: str, texts: list[str]) -> str:
    """按角色摘取区域文字：标题类全取，正文类取前 25 字，其余全取。"""
    joined = " ".join(text.strip() for text in texts if text and text.strip())
    if label in _BODY_LABELS and len(joined) > _BODY_EXCERPT_LIMIT:
        return joined[:_BODY_EXCERPT_LIMIT] + "…"
    return joined


def build_layout_summary(
    regions: list[dict],
    assigned: dict[int, list[int]],
    items: list[dict],
    image_size,
) -> str:
    """生成纯规则、摘取式的中文版面总结（零幻觉句式，对齐榜单示例）。

    regions 为空时返回空字符串；单区域用"页面文字集中在一个主要区域"；
    多区域按"主要分为 N 个区域：方位+角色+内容摘取；逐区域"展开，核心
    区域取归属文字总量最大的 region。
    """
    if not regions:
        return ""

    def describe(region_index: int) -> str:
        region = regions[region_index]
        texts = [
            str(items[member].get("text") or "")
            for member in assigned.get(region_index, [])
        ]
        position = _region_position(region.get("bbox") or (0, 0, 0, 0), image_size)
        role = _region_role_zh(str(region.get("label") or ""))
        excerpt = _region_excerpt(str(region.get("label") or ""), texts)
        prefix = f"{position}{role}" if position else role
        if excerpt:
            return f"{prefix}，内容为“{excerpt}”"
        return prefix

    if len(regions) == 1:
        return f"页面文字集中在一个主要区域：{describe(0)}。"

    descriptions = [describe(index) for index in range(len(regions))]

    def region_text_length(region_index: int) -> int:
        return sum(
            len(str(items[member].get("text") or ""))
            for member in assigned.get(region_index, [])
        )

    # 核心信息只在正文类区域里选，避免“集中在页脚”这类不合理结论
    body_indices = [
        index for index, region in enumerate(regions)
        if str(region.get("label") or "") in _BODY_LABELS
    ]
    core = max(body_indices or range(len(regions)), key=region_text_length)
    core_position = _region_position(
        regions[core].get("bbox") or (0, 0, 0, 0), image_size
    )
    core_role = _region_role_zh(str(regions[core].get("label") or ""))
    core_desc = f"{core_position}{core_role}" if core_position else core_role

    assigned_count = sum(len(members) for members in assigned.values())
    orphan = max(0, len(items) - assigned_count)
    if orphan:
        aux = f"另有{orphan}处文字未归属到具体区域"
    else:
        aux = "全部文字均已归属到对应区域"

    return (
        f"页面文字主要分为{len(regions)}个区域："
        + "；".join(descriptions)
        + f"。页面的核心信息集中在{core_desc}，{aux}。"
    )
