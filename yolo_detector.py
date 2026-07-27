"""
yolo_detector.py

Production-ready YOLOv8 object detector for Raspberry Pi 4, using
ONNX Runtime (CPUExecutionProvider) instead of PyTorch/Ultralytics.

Designed for integration with DonkeyCar as a "part": feed it camera
frames, get back structured Detection objects plus an annotated image,
and use the has_stop_sign() / has_human() / has_obstacle() helpers to
drive stop/avoidance logic.

Model: /home/mishr/mycar/models/best.onnx
Input tensor:  "images"   shape (1, 3, 640, 640)
Output tensor: "output0"  shape (1, 10, 8400)
    -> 4 box coordinates + 6 class scores, per anchor.

No torch / torchvision / ultralytics / triton imports are used.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Class configuration
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = [
    "human",
    "Cone",
    "Safety-barrier",
    "Safety-bollard",
    "Safety-cone",
    "stop_sign",
]

STOP_SIGN_CLASS: str = "stop_sign"
HUMAN_CLASS: str = "human"

OBSTACLE_CLASSES: Tuple[str, ...] = (
    "Cone",
    "Safety-barrier",
    "Safety-bollard",
    "Safety-cone",
)

# BGR colors per class, used by draw_detections().
CLASS_COLORS = {
    "human": (0, 0, 255),
    "Cone": (0, 165, 255),
    "Safety-barrier": (0, 255, 255),
    "Safety-bollard": (255, 0, 255),
    "Safety-cone": (255, 128, 0),
    "stop_sign": (0, 0, 139),
}
DEFAULT_COLOR: Tuple[int, int, int] = (0, 255, 0)


# ---------------------------------------------------------------------------
# Detection data structure
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    A single object detection result, in original-image pixel coordinates.

    Attributes
    ----------
    class_id : int
        Index of the detected class within CLASS_NAMES.
    class_name : str
        Human-readable class name.
    confidence : float
        Detection confidence score in the range [0.0, 1.0].
    x1, y1, x2, y2 : int
        Bounding box corners in original image pixel coordinates
        (top-left and bottom-right).
    """

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Bounding box width in pixels."""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """Bounding box height in pixels."""
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Bounding box center point (cx, cy) in pixel coordinates."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


# ---------------------------------------------------------------------------
# Letterbox preprocessing helper
# ---------------------------------------------------------------------------

@dataclass
class _LetterboxInfo:
    """
    Internal bookkeeping for reversing letterbox preprocessing when
    rescaling predicted boxes back to the original image size.
    """

    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


def _letterbox(
    image: np.ndarray,
    target_size: int = 640,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, _LetterboxInfo]:
    """
    Resize an image to a square (target_size x target_size) canvas while
    preserving aspect ratio, padding the shorter side with pad_color.

    Parameters
    ----------
    image : np.ndarray
        Source image in BGR or RGB order, shape (H, W, 3).
    target_size : int
        Target square size expected by the model (e.g. 640).
    pad_color : tuple of int
        Padding color to fill the letterbox borders with.

    Returns
    -------
    padded : np.ndarray
        The letterboxed square image, shape (target_size, target_size, 3).
    info : _LetterboxInfo
        Scale and padding metadata needed to map boxes back to the
        original image coordinate space.
    """
    original_height, original_width = image.shape[:2]

    scale = min(
        target_size / original_height,
        target_size / original_width,
    )

    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    pad_x = (target_size - new_width) / 2.0
    pad_y = (target_size - new_height) / 2.0

    top = int(round(pad_y - 0.1))
    bottom = target_size - new_height - top
    left = int(round(pad_x - 0.1))
    right = target_size - new_width - left

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    info = _LetterboxInfo(
        scale=scale,
        pad_x=left,
        pad_y=top,
        original_width=original_width,
        original_height=original_height,
    )

    return padded, info


def _scale_boxes_to_original(
    boxes_xyxy: np.ndarray, info: _LetterboxInfo
) -> np.ndarray:
    """
    Map bounding boxes from letterboxed model-input coordinates back to
    the original image coordinate space.

    Parameters
    ----------
    boxes_xyxy : np.ndarray
        Array of shape (N, 4) with columns (x1, y1, x2, y2) in
        letterboxed image coordinates.
    info : _LetterboxInfo
        Letterbox metadata produced by _letterbox().

    Returns
    -------
    np.ndarray
        Array of shape (N, 4), boxes in original image pixel
        coordinates, clipped to the original image bounds.
    """
    boxes = boxes_xyxy.copy()

    boxes[:, [0, 2]] -= info.pad_x
    boxes[:, [1, 3]] -= info.pad_y

    boxes[:, [0, 2]] /= info.scale
    boxes[:, [1, 3]] /= info.scale

    boxes[:, 0] = np.clip(boxes[:, 0], 0, info.original_width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, info.original_height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, info.original_width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, info.original_height - 1)

    return boxes


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    YOLOv8 object detector running on ONNX Runtime, tuned for CPU
    inference on a Raspberry Pi 4.

    This class loads a YOLOv8 model exported to ONNX (output layout
    (1, 10, 8400): 4 box coordinates + 6 class scores per anchor),
    performs letterbox preprocessing, runs inference through
    onnxruntime, decodes predictions, applies confidence filtering and
    non-max suppression, and rescales boxes back to the original image.

    Parameters
    ----------
    model_path : str
        Filesystem path to the .onnx model file.
    confidence_threshold : float
        Minimum class confidence required to keep a raw prediction
        before NMS.
    iou_threshold : float
        IoU threshold used by cv2.dnn.NMSBoxes for suppression of
        overlapping boxes.
    input_size : int
        Square input resolution expected by the model (default 640,
        matching the (1, 3, 640, 640) input tensor).

    Attributes
    ----------
    session : onnxruntime.InferenceSession
        The underlying ONNX Runtime inference session.
    input_name : str
        Name of the model's input tensor ("images").
    output_name : str
        Name of the model's output tensor ("output0").
    last_inference_time : float
        Wall-clock seconds taken by the most recent detect() call's
        inference step. Useful for on-Pi performance tuning.
    """

    def __init__(
        self,
        model_path: str = "/home/mishr/mycar/models/best.onnx",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        """
        Load the ONNX model and prepare the CPU inference session.

        Raises
        ------
        FileNotFoundError
            If model_path does not point to a readable file.
        RuntimeError
            If the ONNX Runtime session fails to initialize.
        """
        self.model_path: str = model_path
        self.confidence_threshold: float = confidence_threshold
        self.iou_threshold: float = iou_threshold
        self.input_size: int = input_size
        self.num_classes: int = len(CLASS_NAMES)
        self.last_inference_time: float = 0.0

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ONNX model from '{model_path}': {exc}"
            ) from exc

        model_inputs = self.session.get_inputs()
        model_outputs = self.session.get_outputs()

        if not model_inputs or not model_outputs:
            raise RuntimeError(
                "ONNX model does not expose expected input/output tensors."
            )

        self.input_name: str = model_inputs[0].name
        self.output_name: str = model_outputs[0].name

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, _LetterboxInfo]:
        """
        Convert a raw BGR image into the normalized, batched NCHW
        float32 tensor expected by the model.

        Parameters
        ----------
        image : np.ndarray
            Source image, shape (H, W, 3), assumed BGR as produced by
            cv2.imread / cv2.VideoCapture.

        Returns
        -------
        input_tensor : np.ndarray
            Float32 array of shape (1, 3, input_size, input_size),
            normalized to [0, 1].
        info : _LetterboxInfo
            Letterbox metadata for reversing preprocessing on output
            boxes.
        """
        letterboxed, info = _letterbox(image, self.input_size)

        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

        normalized = rgb.astype(np.float32) / 255.0

        chw = np.transpose(normalized, (2, 0, 1))

        batched = np.expand_dims(chw, axis=0).astype(np.float32)

        return np.ascontiguousarray(batched), info

    # ------------------------------------------------------------------
    # Output decoding
    # ------------------------------------------------------------------
    def _decode_output(
        self, raw_output: np.ndarray, info: _LetterboxInfo
    ) -> List[Detection]:
        """
        Decode raw model output of shape (1, 10, 8400) into a filtered,
        NMS-applied list of Detection objects in original image
        coordinates.

        Parameters
        ----------
        raw_output : np.ndarray
            Raw output tensor from the ONNX Runtime session, shape
            (1, 4 + num_classes, num_anchors).
        info : _LetterboxInfo
            Letterbox metadata used to rescale boxes back to the
            original image.

        Returns
        -------
        List[Detection]
            Final detections after confidence filtering and NMS.
        """
        predictions = np.squeeze(raw_output, axis=0)
        predictions = predictions.transpose(1, 0)

        box_coords = predictions[:, :4]
        class_scores = predictions[:, 4:4 + self.num_classes]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep_mask = confidences >= self.confidence_threshold
        if not np.any(keep_mask):
            return []

        box_coords = box_coords[keep_mask]
        class_ids = class_ids[keep_mask]
        confidences = confidences[keep_mask]

        cx = box_coords[:, 0]
        cy = box_coords[:, 1]
        w = box_coords[:, 2]
        h = box_coords[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        boxes_xyxy = _scale_boxes_to_original(boxes_xyxy, info)

        nms_boxes = []
        for box in boxes_xyxy:
            bx1, by1, bx2, by2 = box
            nms_boxes.append(
                [float(bx1), float(by1), float(bx2 - bx1), float(by2 - by1)]
            )

        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=confidences.tolist(),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.iou_threshold,
        )

        detections: List[Detection] = []

        if len(indices) == 0:
            return detections

        flat_indices = np.array(indices).flatten()

        for idx in flat_indices:
            idx = int(idx)
            class_id = int(class_ids[idx])
            if 0 <= class_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_id]
            else:
                class_name = f"class_{class_id}"

            bx1, by1, bx2, by2 = boxes_xyxy[idx]

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidences[idx]),
                    x1=int(round(bx1)),
                    y1=int(round(by1)),
                    x2=int(round(bx2)),
                    y2=int(round(by2)),
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------
    def detect(
        self, image: np.ndarray
    ) -> Tuple[List[Detection], np.ndarray]:
        """
        Run full detection on a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            Source image, shape (H, W, 3), BGR color order (as
            produced by cv2.imread / cv2.VideoCapture, or DonkeyCar's
            camera part after BGR conversion). If your pipeline
            supplies RGB frames, convert to BGR before calling, or
            adapt _preprocess accordingly.

        Returns
        -------
        detections : List[Detection]
            All detections surviving confidence filtering and NMS.
        annotated_image : np.ndarray
            A copy of the input image with bounding boxes and labels
            drawn on it.

        Raises
        ------
        ValueError
            If image is None or not a valid 3-channel array.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "detect() requires a non-empty HxWx3 image array."
            )

        input_tensor, info = self._preprocess(image)

        start_time = time.perf_counter()
        raw_output = self.session.run(
            [self.output_name], {self.input_name: input_tensor}
        )[0]
        self.last_inference_time = time.perf_counter() - start_time

        detections = self._decode_output(raw_output, info)

        annotated_image = self.draw_detections(image, detections)

        return detections, annotated_image

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_detections(
        self, image: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """
        Draw bounding boxes, class labels, and confidence scores onto
        a copy of the input image.

        Parameters
        ----------
        image : np.ndarray
            Source image, shape (H, W, 3).
        detections : List[Detection]
            Detections to render.

        Returns
        -------
        np.ndarray
            A new image array (the input is not modified in place)
            with annotations drawn.
        """
        annotated = image.copy()

        for det in detections:
            color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)

            cv2.rectangle(
                annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2
            )

            label = f"{det.class_name} {det.confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            label_top = max(det.y1 - text_height - baseline - 4, 0)

            cv2.rectangle(
                annotated,
                (det.x1, label_top),
                (det.x1 + text_width + 4, label_top + text_height + baseline + 4),
                color,
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (det.x1 + 2, label_top + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def shutdown(self) -> None:
        """Release the ONNX Runtime session. Safe to call multiple times."""
        self.session = None


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------

def has_stop_sign(
    detections: List[Detection], min_confidence: float = 0.0
) -> bool:
    """
    Return True if any detection in the list is a stop sign.

    Parameters
    ----------
    detections : List[Detection]
        Detections to inspect.
    min_confidence : float
        Optional additional confidence floor applied on top of
        whatever threshold the detector already used.

    Returns
    -------
    bool
    """
    return any(
        det.class_name == STOP_SIGN_CLASS and det.confidence >= min_confidence
        for det in detections
    )


def has_human(
    detections: List[Detection], min_confidence: float = 0.0
) -> bool:
    """
    Return True if any detection in the list is a human.

    Parameters
    ----------
    detections : List[Detection]
        Detections to inspect.
    min_confidence : float
        Optional additional confidence floor.

    Returns
    -------
    bool
    """
    return any(
        det.class_name == HUMAN_CLASS and det.confidence >= min_confidence
        for det in detections
    )


def has_obstacle(
    detections: List[Detection], min_confidence: float = 0.0
) -> bool:
    """
    Return True if any detection in the list is a non-human,
    non-stop-sign obstacle (Cone, Safety-barrier, Safety-bollard, or
    Safety-cone).

    Parameters
    ----------
    detections : List[Detection]
        Detections to inspect.
    min_confidence : float
        Optional additional confidence floor.

    Returns
    -------
    bool
    """
    return any(
        det.class_name in OBSTACLE_CLASSES and det.confidence >= min_confidence
        for det in detections
    )


def get_detection_by_name(
    detections: List[Detection], class_name: str
) -> List[Detection]:
    """
    Return all detections matching a given class name.

    Parameters
    ----------
    detections : List[Detection]
        Detections to filter.
    class_name : str
        Class name to match, e.g. "stop_sign".

    Returns
    -------
    List[Detection]
        Detections whose class_name equals the given class_name. Empty
        list if none match.
    """
    return [det for det in detections if det.class_name == class_name]


def count_objects(detections: List[Detection]) -> dict:
    """
    Count detections per class name.

    Parameters
    ----------
    detections : List[Detection]
        Detections to count.

    Returns
    -------
    dict
        Mapping of class_name -> count, for classes that appear at
        least once. Classes with zero detections are omitted.
    """
    counts: dict = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return counts


def nearest_detection(
    detections: List[Detection], image_height: int
) -> Optional[Detection]:
    """
    Return the detection whose bounding box is closest to the bottom
    of the frame (a simple proxy for "nearest to the camera" when the
    camera is mounted looking forward/down on a ground vehicle).

    Parameters
    ----------
    detections : List[Detection]
        Detections to search.
    image_height : int
        Height in pixels of the source image, used as the reference
        for "closest to the bottom of the frame".

    Returns
    -------
    Optional[Detection]
        The detection with the largest y2 (closest to the bottom of
        the frame), or None if detections is empty.
    """
    if not detections:
        return None

    return max(detections, key=lambda det: det.y2)


def largest_detection(detections: List[Detection]) -> Optional[Detection]:
    """
    Return the detection with the largest bounding box area.

    Parameters
    ----------
    detections : List[Detection]
        Detections to search.

    Returns
    -------
    Optional[Detection]
        The detection with the greatest area, or None if detections
        is empty.
    """
    if not detections:
        return None

    return max(detections, key=lambda det: det.area)


# ---------------------------------------------------------------------------
# DonkeyCar part wrapper
# ---------------------------------------------------------------------------

class YoloDetectorPart:
    """
    Thin DonkeyCar part wrapper around YOLODetector, exposing a run()
    method compatible with donkeycar.vehicle.Vehicle.add().

    Outputs, in order: (annotated_image, detections, has_stop_sign,
    has_human, has_obstacle).
    """

    def __init__(
        self,
        model_path: str = "/home/mishr/mycar/models/best.onnx",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        """Initialize the underlying YOLODetector instance."""
        self.detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size,
        )
        self.detections: List[Detection] = []

    def run(
        self, img_arr: np.ndarray
    ) -> Tuple[np.ndarray, List[Detection], bool, bool, bool]:
        """
        DonkeyCar part entry point, called once per camera frame.

        Parameters
        ----------
        img_arr : np.ndarray
            Camera frame, shape (H, W, 3), BGR order.

        Returns
        -------
        Tuple[np.ndarray, List[Detection], bool, bool, bool]
            (annotated_image, detections, has_stop_sign_flag,
            has_human_flag, has_obstacle_flag)
        """
        if img_arr is None:
            return img_arr, [], False, False, False

        detections, annotated_image = self.detector.detect(img_arr)
        self.detections = detections

        return (
            annotated_image,
            detections,
            has_stop_sign(detections),
            has_human(detections),
            has_obstacle(detections),
        )

    def shutdown(self) -> None:
        """Release detector resources on vehicle shutdown."""
        self.detector.shutdown()
