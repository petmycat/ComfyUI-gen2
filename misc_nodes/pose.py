"""
Gen2 Pose - DWPose detector utilities and ComfyUI node
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import warnings

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Callable, Union, Optional
from timeit import default_timer
from huggingface_hub import hf_hub_download

import comfy.model_management as model_management

from custom_nodes.comfyui_controlnet_aux.utils import common_annotator_call, define_preprocessor_inputs, INPUT
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose import util
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.body import Body, BodyResult, Keypoint
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.hand import Hand
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.face import Face
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.types import PoseResult, HandResult, FaceResult, BodyResult, Keypoint
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.wholebody import Wholebody
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.util import HWC3, resize_image_with_pad, common_input_validate, custom_hf_download

from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.dw_onnx.cv_ox_det import inference_detector as inference_onnx_yolox
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.dw_onnx.cv_ox_yolo_nas import inference_detector as inference_onnx_yolo_nas
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.dw_onnx.cv_ox_pose import inference_pose as inference_onnx_pose

from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.dw_torchscript.jit_det import inference_detector as inference_jit_yolox
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.dw_torchscript.jit_pose import inference_pose as inference_jit_pose

from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose.util import guess_onnx_input_shape_dtype, get_model_type, get_ort_providers, is_model_torchscript


# =============================================================================
# ONNXRUNTIME GPU CHECK
# =============================================================================

DWPOSE_MODEL_NAME = "yzd-v/DWPose"

GPU_PROVIDERS = ["CUDAExecutionProvider", "DirectMLExecutionProvider", "OpenVINOExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider"]

def check_ort_gpu():
    try:
        import onnxruntime as ort
        for provider in GPU_PROVIDERS:
            if provider in ort.get_available_providers():
                return True
        return False
    except:
        return False

if not os.environ.get("DWPOSE_ONNXRT_CHECKED"):
    if check_ort_gpu():
        print("DWPose: Onnxruntime with acceleration providers detected")
    else:
        warnings.warn("DWPose: Onnxruntime not found or doesn't come with acceleration providers, switch to OpenCV with CPU device. DWPose might run very slowly")
        os.environ['AUX_ORT_PROVIDERS'] = ''
    os.environ["DWPOSE_ONNXRT_CHECKED"] = '1'


# =============================================================================
# POSE DRAWING / ENCODING UTILITIES
# =============================================================================

def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True, xinsr_stick_scaling=False):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints, xinsr_stick_scaling)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas


def decode_json_as_poses(
    pose_json: dict,
) -> Tuple[List[PoseResult], int, int]:
    """Decode the json_string complying with the openpose JSON output format
    to poses that controlnet recognizes.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

    Args:
        json_string: The json string to decode.

    Returns:
        human_poses
        canvas_height
        canvas_width
    """
    height = pose_json["canvas_height"]
    width = pose_json["canvas_width"]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def decompress_keypoints(
        numbers: Optional[List[float]],
    ) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None

        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            if c < 1.0:
                return None
            keypoint = Keypoint(x, y)
            return keypoint

        return [create_keypoint(x, y, c) for x, y, c in chunks(numbers, n=3)]

    return (
        [
            PoseResult(
                body=BodyResult(
                    keypoints=decompress_keypoints(pose.get("pose_keypoints_2d"))
                ),
                left_hand=decompress_keypoints(pose.get("hand_left_keypoints_2d")),
                right_hand=decompress_keypoints(pose.get("hand_right_keypoints_2d")),
                face=decompress_keypoints(pose.get("face_keypoints_2d")),
            )
            for pose in pose_json.get("people", [])
        ],
        height,
        width,
    )


def encode_poses_as_dict(poses: List[PoseResult], canvas_height: int, canvas_width: int) -> str:
    """ Encode the pose as a dict following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """
    def compress_keypoints(keypoints: Union[List[Keypoint], None]) -> Union[List[float], None]:
        if not keypoints:
            return None

        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return {
        'people': [
            {
                'pose_keypoints_2d': compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d":compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        'canvas_height': canvas_height,
        'canvas_width': canvas_width,
    }


# =============================================================================
# DWPOSE DETECTOR CLASSES
# =============================================================================

class thWholebody(Wholebody):
    def __init__(self, det_model_path: Optional[str] = None, pose_model_path: Optional[str] = None, torchscript_device="cuda"):
        self.det_filename = det_model_path and os.path.basename(det_model_path)
        self.pose_filename = pose_model_path and os.path.basename(pose_model_path)
        self.det, self.pose = None, None
        # return type: None ort cv2 torchscript
        self.det_model_type = get_model_type("DWPose",self.det_filename)
        self.pose_model_type = get_model_type("DWPose",self.pose_filename)
        # Always loads to CPU to avoid building OpenCV.
        cv2_device = 'cpu'
        cv2_backend = cv2.dnn.DNN_BACKEND_OPENCV if cv2_device == 'cpu' else cv2.dnn.DNN_BACKEND_CUDA
        # You need to manually build OpenCV through cmake to work with your GPU.
        cv2_providers = cv2.dnn.DNN_TARGET_CPU if cv2_device == 'cpu' else cv2.dnn.DNN_TARGET_CUDA
        ort_providers = get_ort_providers()

        if self.det_model_type is None:
            pass
        elif self.det_model_type == "ort":
            try:
                import onnxruntime as ort
                self.det = ort.InferenceSession(det_model_path, providers=ort_providers)
            except:
                print(f"Failed to load onnxruntime with {self.det.get_providers()}.\nPlease change EP_list in the config.yaml and restart ComfyUI")
                self.det = ort.InferenceSession(det_model_path, providers=["CPUExecutionProvider"])
        elif self.det_model_type == "cv2":
            try:
                self.det = cv2.dnn.readNetFromONNX(det_model_path)
                self.det.setPreferableBackend(cv2_backend)
                self.det.setPreferableTarget(cv2_providers)
            except:
                print("TopK operators may not work on your OpenCV, try use onnxruntime with CPUExecutionProvider")
                try:
                    import onnxruntime as ort
                    self.det = ort.InferenceSession(det_model_path, providers=["CPUExecutionProvider"])
                except:
                    print(f"Failed to load {det_model_path}, you can use other models instead")
        else:
            self.det = torch.jit.load(det_model_path)
            self.det.to(torchscript_device)

        if self.pose_model_type is None:
            pass
        elif self.pose_model_type == "ort":
            try:
                import onnxruntime as ort
                self.pose = ort.InferenceSession(pose_model_path, providers=ort_providers)
            except:
                print(f"Failed to load onnxruntime with {self.pose.get_providers()}.\nPlease change EP_list in the config.yaml and restart ComfyUI")
                self.pose = ort.InferenceSession(pose_model_path, providers=["CPUExecutionProvider"])
        elif self.pose_model_type == "cv2":
            self.pose = cv2.dnn.readNetFromONNX(pose_model_path)
            self.pose.setPreferableBackend(cv2_backend)
            self.pose.setPreferableTarget(cv2_providers)
        else:
            self.pose = torch.jit.load(pose_model_path)
            self.pose.to(torchscript_device)

        if self.pose_filename is not None:
            self.pose_input_size, _ = guess_onnx_input_shape_dtype(self.pose_filename)

    def __call__(self, oriImg, threshold) -> Optional[np.ndarray]:
        #Sacrifice accurate time measurement for compatibility

        det_result = None

        if self.det is None:
            print("DWPose: No detector specified, using full image for pose estimation.") # pragma: no cover
            det_result = []
        else:
            det_start = default_timer()
            if is_model_torchscript(self.det):
                det_result = inference_jit_yolox(self.det, oriImg, detect_classes=[0])
            else:
                if "yolox" in self.det_filename:
                    det_result = inference_onnx_yolox(self.det, oriImg, detect_classes=[0], dtype=np.float32)
                else:
                    #FP16 and INT8 YOLO NAS accept uint8 input
                    det_result = inference_onnx_yolo_nas(self.det, oriImg, detect_classes=[0], dtype=np.uint8)
            print(f"DWPose: Bbox {((default_timer() - det_start) * 1000):.2f}ms")
            if (det_result is None) or (det_result.shape[0] == 0):
                return None

        pose_start = default_timer()
        if is_model_torchscript(self.pose):
            keypoints, scores = inference_jit_pose(self.pose, det_result, oriImg, self.pose_input_size)
        else:
            _, pose_onnx_dtype = guess_onnx_input_shape_dtype(self.pose_filename)
            keypoints, scores = inference_onnx_pose(self.pose, det_result, oriImg, self.pose_input_size, dtype=pose_onnx_dtype)

        num_subjects_log = 'full image'
        if hasattr(det_result, 'shape') and det_result.shape[0] > 0:
            num_subjects_log = f"{det_result.shape[0]} people"
        elif isinstance(det_result, list) and len(det_result) > 0 and isinstance(det_result[0], (list, np.ndarray)):
             num_subjects_log = f"{len(det_result)} people"

        print(f"DWPose: Pose {((default_timer() - pose_start) * 1000):.2f}ms on {num_subjects_log}\n")

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > threshold,
            keypoints_info[:, 6, 2:4] > threshold).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        return keypoints_info

    @staticmethod
    def format_result(keypoints_info: Optional[np.ndarray]) -> List[PoseResult]:
        def format_keypoint_part(
            part: np.ndarray,
        ) -> Optional[List[Optional[Keypoint]]]:
            keypoints = [
                Keypoint(x, y, score, i) if score >= 0.3 else None
                for i, (x, y, score) in enumerate(part)
            ]
            return (
                None if all(keypoint is None for keypoint in keypoints) else keypoints
            )

        def total_score(keypoints: Optional[List[Optional[Keypoint]]]) -> float:
            return (
                sum(keypoint.score for keypoint in keypoints if keypoint is not None)
                if keypoints is not None
                else 0.0
            )

        pose_results = []
        if keypoints_info is None:
            return pose_results

        for instance in keypoints_info:
            body_keypoints = format_keypoint_part(instance[:18]) or ([None] * 18)
            left_hand = format_keypoint_part(instance[92:113])
            right_hand = format_keypoint_part(instance[113:134])
            face = format_keypoint_part(instance[24:92])

            # Openpose face consists of 70 points in total, while DWPose only
            # provides 68 points. Padding the last 2 points.
            if face is not None:
                # left eye
                face.append(body_keypoints[14])
                # right eye
                face.append(body_keypoints[15])

            body = BodyResult(
                body_keypoints, total_score(body_keypoints), len(body_keypoints)
            )
            pose_results.append(PoseResult(body, left_hand, right_hand, face))

        return pose_results


global_cached_dwpose = thWholebody()


class DwposeDetector:
    """
    A class for detecting human poses in images using the Dwpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    def __init__(self, dw_pose_estimation):
        self.dw_pose_estimation = dw_pose_estimation

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, pretrained_det_model_or_path=None, det_filename=None, pose_filename=None, torchscript_device="cuda"):
        global global_cached_dwpose
        pretrained_det_model_or_path = pretrained_det_model_or_path or pretrained_model_or_path

        pose_filename = pose_filename or "dw-ll_ucoco_384.onnx"

        det_model_path = None
        if det_filename is not None:
            det_model_path = custom_hf_download(pretrained_det_model_or_path, det_filename)
        pose_model_path = custom_hf_download(pretrained_model_or_path, pose_filename)

        print(f"\nDWPose: Using {det_filename} for bbox detection and {pose_filename} for pose estimation")
        if global_cached_dwpose.det is None or global_cached_dwpose.det_filename != det_filename:
            t = thWholebody(det_model_path, None, torchscript_device=torchscript_device)
            t.pose = global_cached_dwpose.pose
            t.pose_filename = global_cached_dwpose.pose
            global_cached_dwpose = t

        if global_cached_dwpose.pose is None or global_cached_dwpose.pose_filename != pose_filename:
            t = thWholebody(None, pose_model_path, torchscript_device=torchscript_device)
            t.det = global_cached_dwpose.det
            t.det_filename = global_cached_dwpose.det_filename
            global_cached_dwpose = t
        return cls(global_cached_dwpose)

    def detect_poses(self, oriImg, threshold) -> List[PoseResult]:
        with torch.no_grad():
            keypoints_info = self.dw_pose_estimation(oriImg.copy(), threshold)
            return thWholebody.format_result(keypoints_info)

    def __call__(self, input_image, threshold, detect_resolution=512, include_body=True, include_hand=False, include_face=False, hand_and_face=None, output_type="pil", image_and_json=False, upscale_method="INTER_CUBIC", xinsr_stick_scaling=False, **kwargs):
        if hand_and_face is not None:
            warnings.warn("hand_and_face is deprecated. Use include_hand and include_face instead.", DeprecationWarning)
            include_hand = hand_and_face
            include_face = hand_and_face

        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, _ = resize_image_with_pad(input_image, 0, upscale_method)
        poses = self.detect_poses(input_image, threshold)

        canvas = draw_poses(poses, input_image.shape[0], input_image.shape[1], draw_body=include_body, draw_hand=include_hand, draw_face=include_face, xinsr_stick_scaling=xinsr_stick_scaling)
        canvas, remove_pad = resize_image_with_pad(canvas, detect_resolution, upscale_method)
        detected_map = HWC3(remove_pad(canvas))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        if image_and_json:
            return (detected_map, encode_poses_as_dict(poses, input_image.shape[0], input_image.shape[1]))

        return detected_map


# =============================================================================
# COMFYUI NODE
# =============================================================================

class Gen2_DwposeThreshold:
    """
    Custom DWPose node with a confidence threshold slider.
    """

    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            detect_hand=INPUT.COMBO(["enable", "disable"]),
            detect_body=INPUT.COMBO(["enable", "disable"]),
            detect_face=INPUT.COMBO(["enable", "disable"]),
            threshold=INPUT.FLOAT(),
            resolution=INPUT.RESOLUTION(),
            bbox_detector=INPUT.COMBO(
                ["None"] + ["yolox_l.torchscript.pt", "yolox_l.onnx", "yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx"],
                default="yolox_l.onnx"
            ),
            pose_estimator=INPUT.COMBO(
                ["dw-ll_ucoco_384_bs5.torchscript.pt", "dw-ll_ucoco_384.onnx", "dw-ll_ucoco.onnx"],
                default="dw-ll_ucoco_384_bs5.torchscript.pt"
            ),
            scale_stick_for_xinsr_cn=INPUT.COMBO(["disable", "enable"])
        )

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "Gen2/Pose"

    def estimate_pose(self, image, detect_hand="enable", detect_body="enable", detect_face="enable", threshold=0.30, resolution=512, bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx", scale_stick_for_xinsr_cn="disable", **kwargs):
        if bbox_detector == "None":
            yolo_repo = DWPOSE_MODEL_NAME
        elif bbox_detector == "yolox_l.onnx":
            yolo_repo = DWPOSE_MODEL_NAME
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        else:
            raise NotImplementedError(f"Download mechanism for {bbox_detector}")

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = DWPOSE_MODEL_NAME
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            raise NotImplementedError(f"Download mechanism for {pose_estimator}")

        model = DwposeDetector.from_pretrained(
            pose_repo,
            yolo_repo,
            det_filename=(None if bbox_detector == "None" else bbox_detector), pose_filename=pose_estimator,
            torchscript_device=model_management.get_torch_device()
        )
        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"
        scale_stick_for_xinsr_cn = scale_stick_for_xinsr_cn == "enable"
        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = model(image, **kwargs)
            self.openpose_dicts.append(openpose_dict)
            return pose_img

        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, threshold=threshold, resolution=resolution, xinsr_stick_scaling=scale_stick_for_xinsr_cn)
        del model
        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts)
        }


# =============================================================================
# NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "Gen2_DwposeThreshold": Gen2_DwposeThreshold,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_DwposeThreshold": "Gen2 DWpose with Threshold",
}

