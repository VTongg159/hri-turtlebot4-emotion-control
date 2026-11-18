"""Facial-expression backends behind one interface.

``analyze(frame_bgr)`` returns a list of ``FaceResult`` (possibly empty), each
holding a bounding box and scores over the six target emotions (summing to 1).

* ``deepface``: DeepFace.analyze(actions=["emotion"]); default.
* ``fer``: the FER package (Keras CNN + OpenCV Haar or MTCNN detector).
* ``ferplus``: FER+ ONNX model on OpenCV DNN; needs no TensorFlow.
* ``hsemotion``: HSEmotion EfficientNet-B0 (AffectNet) via onnxruntime; most
  accurate on a real webcam.
* ``auto``: hsemotion if available, else deepface, else ferplus.
* ``none``: face detection only (OpenCV Haar), every face scored neutral.
  Used for testing the camera and UI without TensorFlow, together with the
  manual keyboard override.

Heavy libraries are imported lazily so ``--help`` and unit tests stay fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .emotions import EMOTIONS, normalize_scores


@dataclass
class FaceResult:
    box: tuple[int, int, int, int]  # x, y, w, h in pixels
    scores: dict  # emotion -> probability over EMOTIONS

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


def haar_cascade_path(name: str = "haarcascade_frontalface_default.xml") -> str:
    """pip opencv-python ships cv2.data; Ubuntu's python3-opencv uses opencv-data."""
    import os  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    candidates = []
    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades, name))
    candidates += [f"/usr/share/opencv4/haarcascades/{name}", f"/usr/share/opencv/haarcascades/{name}"]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{name} not found (pip install opencv-python or apt install opencv-data)")


class Backend:
    name = "base"

    def warmup(self) -> None:
        """Run one inference so model download/compile happens before the loop."""
        self.analyze(np.zeros((240, 320, 3), dtype=np.uint8))

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        raise NotImplementedError


class DeepFaceBackend(Backend):
    name = "deepface"

    def __init__(self, detector_backend: str = "opencv", min_face_confidence: float = 0.5):
        from deepface import DeepFace  # noqa: PLC0415  (lazy, heavy import)

        self._deepface = DeepFace
        self.detector_backend = detector_backend
        self.min_face_confidence = min_face_confidence

    def warmup(self) -> None:
        # Force the emotion CNN to load now (analyze() skips it when no face is found).
        self._deepface.analyze(img_path=np.zeros((240, 320, 3), dtype=np.uint8), actions=["emotion"],
                               detector_backend=self.detector_backend, enforce_detection=False, silent=True)

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        try:
            # enforce_detection=True: without a face DeepFace raises before running
            # the emotion CNN, instead of classifying the whole image.
            results = self._deepface.analyze(
                img_path=frame_bgr,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=True,
                silent=True,
            )
        except ValueError:  # "Face could not be detected"
            return []
        if isinstance(results, dict):
            results = [results]

        faces = []
        for r in results:
            region = r.get("region", {}) or {}
            box = tuple(int(region.get(k, 0)) for k in ("x", "y", "w", "h"))
            confidence = float(r.get("face_confidence", 1.0) or 0.0)
            if box[2] <= 0 or confidence < self.min_face_confidence:
                continue
            faces.append(FaceResult(box, normalize_scores(r.get("emotion", {}))))
        return faces


class FerBackend(Backend):
    name = "fer"

    def __init__(self, use_mtcnn: bool = False):
        try:
            from fer import FER  # noqa: PLC0415
        except ImportError:  # fer>=25 moved the class
            from fer.fer import FER  # noqa: PLC0415
        self._detector = FER(mtcnn=use_mtcnn)

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        rgb = frame_bgr[:, :, ::-1].copy()
        faces = []
        for r in self._detector.detect_emotions(rgb):
            x, y, w, h = (int(v) for v in r["box"])
            faces.append(FaceResult((x, y, w, h), normalize_scores(r["emotions"])))
        return faces


class HaarOnlyBackend(Backend):
    name = "none"

    def __init__(self) -> None:
        import cv2  # noqa: PLC0415

        self._cv2 = cv2
        self._cascade = cv2.CascadeClassifier(haar_cascade_path())
        if self._cascade.empty():
            raise RuntimeError("cannot load haarcascade_frontalface_default.xml")

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        neutral = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
        return [FaceResult(tuple(int(v) for v in b), dict(neutral)) for b in boxes]


class FerPlusBackend(Backend):
    """FER+ CNN (Microsoft, ONNX model zoo) run by OpenCV DNN, no TensorFlow.

    Haar cascade finds faces; each face is cropped with a small margin,
    converted to 64x64 grayscale and classified into 8 classes. "contempt" and
    "disgust" are dropped by ``normalize_scores``.
    """

    name = "ferplus"
    CLASSES = ("neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt")

    def __init__(self, model_path: str = "") -> None:
        import cv2  # noqa: PLC0415

        self._cv2 = cv2
        path = model_path or default_ferplus_path()
        self._net = cv2.dnn.readNetFromONNX(path)
        self._cascade = cv2.CascadeClassifier(haar_cascade_path())
        if self._cascade.empty():
            raise RuntimeError("cannot load haarcascade_frontalface_default.xml")

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        boxes = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        faces = []
        for x, y, w, h in boxes:
            m = int(0.1 * w)  # FER+ was trained on slightly looser crops
            x0, y0 = max(0, x - m), max(0, y - m)
            x1, y1 = min(W, x + w + m), min(H, y + h + m)
            crop = cv2.resize(gray[y0:y1, x0:x1], (64, 64)).astype(np.float32)
            # test-time augmentation: average the face and its mirror image
            prob = (self._classify(crop) + self._classify(crop[:, ::-1].copy())) / 2.0
            scores = normalize_scores(dict(zip(self.CLASSES, prob.tolist())))
            faces.append(FaceResult((int(x), int(y), int(w), int(h)), scores))
        return faces

    def _classify(self, crop: np.ndarray) -> np.ndarray:
        self._net.setInput(crop.reshape(1, 1, 64, 64))
        return _softmax(self._net.forward().ravel())


class HSEmotionBackend(Backend):
    """HSEmotion EfficientNet-B0 (Savchenko, trained on AffectNet) via onnxruntime.

    AffectNet's in-the-wild faces make it far more robust on a real webcam than
    the 48x48 FER2013 models behind DeepFace and FER. Haar cascade finds faces;
    each crop (8 % margin) is resized to 224x224 RGB with ImageNet
    normalisation, classified together with its mirror image, and the 8 classes
    are reduced to the six target emotions.
    """

    name = "hsemotion"
    MODEL = "enet_b0_8_best_afew.onnx"
    CLASSES = ("anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path: str = "") -> None:
        import cv2  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415

        self._cv2 = cv2
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # leave the CPU to Gazebo
        self._session = ort.InferenceSession(model_path or package_model_path(self.MODEL), opts,
                                             providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self._cascade = cv2.CascadeClassifier(haar_cascade_path())
        if self._cascade.empty():
            raise RuntimeError("cannot load haarcascade_frontalface_default.xml")

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        faces = []
        for x, y, w, h in self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)):
            m = int(0.08 * w)
            crop = frame_bgr[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
            rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0
            rgb = (rgb - self.MEAN) / self.STD
            batch = np.stack([rgb, rgb[:, ::-1]]).transpose(0, 3, 1, 2).astype(np.float32)  # + mirror
            logits = self._session.run(None, {self._input: batch})[0]
            prob = np.mean([_softmax(z) for z in logits], axis=0)
            faces.append(FaceResult((int(x), int(y), int(w), int(h)),
                                    normalize_scores(dict(zip(self.CLASSES, prob.tolist())))))
        return faces


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def package_model_path(name: str) -> str:
    """Locate a bundled model in share/tb4_emotion_control/models (or the source tree)."""
    import os  # noqa: PLC0415

    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

        candidates.append(os.path.join(get_package_share_directory("tb4_emotion_control"), "models", name))
    except Exception:  # not installed / no ROS environment
        pass
    candidates.append(os.path.join(os.path.dirname(__file__), "..", "models", name))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{name} not found; run scripts/download_models.sh")


def default_ferplus_path() -> str:
    return package_model_path("emotion-ferplus-8.onnx")


def hsemotion_available() -> bool:
    import importlib.util  # noqa: PLC0415

    if importlib.util.find_spec("onnxruntime") is None:
        return False
    try:
        package_model_path(HSEmotionBackend.MODEL)
    except FileNotFoundError:
        return False
    return True


def deepface_available() -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("deepface") is not None and (
        importlib.util.find_spec("tensorflow") is not None
    )


_FACTORIES: dict[str, Callable[..., Backend]] = {
    "deepface": DeepFaceBackend,
    "fer": FerBackend,
    "ferplus": FerPlusBackend,
    "hsemotion": HSEmotionBackend,
    "none": HaarOnlyBackend,
}


def resolve_backend_name(name: str) -> str:
    """``auto`` = HSEmotion if available, else DeepFace, else the TensorFlow-free FER+."""
    if name == "auto":
        if hsemotion_available():
            return "hsemotion"
        return "deepface" if deepface_available() else "ferplus"
    return name


def create_backend(name: str, **kwargs) -> Backend:
    name = resolve_backend_name(name)
    try:
        factory = _FACTORIES[name]
    except KeyError:
        choices = sorted(_FACTORIES) + ["auto"]
        raise ValueError(f"unknown backend '{name}', choose from {choices}") from None
    allowed = {
        "deepface": ("detector_backend", "min_face_confidence"),
        "fer": ("use_mtcnn",),
        "ferplus": ("model_path",),
        "hsemotion": ("model_path",),
    }.get(name, ())
    return factory(**{k: v for k, v in kwargs.items() if k in allowed and v not in (None, "")})


def largest_face(faces: list[FaceResult]) -> FaceResult | None:
    """The person closest to the camera is assumed to be the operator."""
    return max(faces, key=lambda f: f.area) if faces else None
