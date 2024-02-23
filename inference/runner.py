import copy
from dataclasses import dataclass
from typing import List, Optional
import uuid
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet3d.core import bbox3d2result

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.runner import load_checkpoint
from mmcv import Config
import numpy as np
from pyquaternion import Quaternion
import torch

from projects.mmdet3d_plugin.VAD.VAD import VAD
from nuscenes.eval.common.utils import (
    quaternion_yaw,
)

NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@dataclass
class VADInferenceInput:
    imgs: np.ndarray
    """shape: (n-cams (6), h (900), w (1600) c (3)) | images without any preprocessing. should be in RGB order as uint8"""
    lidar_pose: np.ndarray
    """shape: (3, 4) | lidar pose in global frame"""
    lidar2img: np.ndarray
    """shape: (n-cams (6), 4, 4) | lidar2img transformation matrix, i.e., lidar2cam @ camera2img"""
    timestamp: float
    """timestamp of the current frame in seconds"""
    can_bus_signals: np.ndarray
    """shape: (16,) | see above for details"""
    command: int
    """0: right, 1: left, 2: straight"""


@dataclass
class VADAuxOutputs:
    objects_in_bev: Optional[List[List[float]]] = None  # N x [x, y, width, height, yaw]
    object_classes: Optional[List[str]] = None  # (N, )
    object_scores: Optional[List[float]] = None  # (N, )
    segmentation: Optional[List[List[float]]] = None
    seg_grid_centers: Optional[
        List[List[List[float]]]
    ] = None  # bev_h (200), bev_w (200), 2 (x & y)
    future_trajs: Optional[List[List[List[List[float]]]]] = None  # (N, 6, 6, 2)

    def to_json(self) -> dict:
        return dict(
            objects_in_bev=self.objects_in_bev,
            object_classes=self.object_classes,
            object_scores=self.object_scores,
            segmentation=self.segmentation,
            seg_grid_centers=self.seg_grid_centers,
            future_trajs=self.future_trajs,
        )


@dataclass
class VADInferenceOutput:
    trajectory: np.ndarray
    """shape: (n-future (6), 2) | predicted trajectory in the ego-frame @ 2Hz"""
    aux_outputs: Optional[VADAuxOutputs] = None
    """aux outputs such as objects, tracks, segmentation and motion forecast"""


class VADRunner:
    def __init__(self, config_path: str, checkpoint_path: str, device: torch.device):
        config = Config.fromfile(config_path)
        self.config = config

        self.model: VAD = build_model(
            config.model, train_cfg=None, test_cfg=config.get("test_cfg")
        )

        self.model.eval()
        # load the checkpoint
        if checkpoint_path is not None:
            ckpt = load_checkpoint(self.model, checkpoint_path, map_location="cpu")
            self.classes = ckpt["meta"]["CLASSES"]
        else:
            raise ValueError("checkpoint_path is None")

        # do more stuff here maybe?
        self.model = self.model.to(device)
        self.device = device
        self.preproc_pipeline = Compose(config.inference_pipeline)
        self.reset()

    def reset(self):
        # making a new scene token for each new scene. these are used in the model.
        self.scene_token = str(uuid.uuid4())
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    def _preproc_canbus(self, input: VADInferenceInput):
        """Preprocesses the raw canbus signals from nuscenes."""
        rotation = Quaternion(input.can_bus_signals[3:7])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        # extend the canbus signals with the patch angle, first in radians then in degrees
        input.can_bus_signals = np.append(
            input.can_bus_signals, patch_angle / 180 * np.pi
        )
        input.can_bus_signals = np.append(input.can_bus_signals, patch_angle)

    def preproc(self, input: VADInferenceInput):
        """Preprocess the input data."""
        self._preproc_canbus(input)
        # TODO: make torch version of the preproc (for images) pipeline instead of using mmcv version'

    @torch.no_grad()
    def forward_inference(self, input: VADInferenceInput) -> VADInferenceOutput:
        """Run inference without all the preprocessed dataset stuff."""
        # permute rgb -> bgr
        imgs = input.imgs[:, :, :, ::-1]
        # input to preproc shoudl be dict(img=imgs) where imgs: n x h x w x c in bgr format
        preproc_input = dict(img=imgs, lidar2img=input.lidar2img)
        # run it through the inference pipeline (which is same as eval pipeline except not loading annotations)
        preproc_output = self.preproc_pipeline(preproc_input)
        # collect in array as will convert to tensor, but currently it is a list of arrays (n, h, w, c)
        imgs = np.array(preproc_output["img"])
        # move back to the nchw format
        imgs = np.moveaxis(imgs, -1, 1)
        # convert to tensor and move to device
        imgs = torch.from_numpy(imgs).to(self.device)
        # img should be (1, n, 3, h, w)
        imgs = imgs.unsqueeze(0)
        # we are preproccessing the canbus signals only currently.
        # TODO: fix preproc to include the image preprocessing as well. this is currently done
        # in mmcv (i.e., numpy) and not torch.
        self.preproc(input)

        # we need to emulate the img_metas here in order to run the model.
        img_metas = [
            {
                "scene_token": self.scene_token,
                "can_bus": input.can_bus_signals,
                "lidar2img": preproc_output["lidar2img"],  # lidar2cam @ camera2img
                "img_shape": preproc_output["img_shape"],
                "box_type_3d": LiDARInstance3DBoxes,
            }
        ]

        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        # first frame
        if self.prev_frame_info["scene_token"] is None:
            img_metas[0]["can_bus"][:3] = 0
            img_metas[0]["can_bus"][-1] = 0
        # following frames
        else:
            img_metas[0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]

        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle

        img_feats = self.model.extract_feat(img=imgs, img_metas=img_metas)
        outs = self.model.pts_bbox_head(
            img_feats,
            img_metas,
            prev_bev=self.prev_frame_info["prev_bev"],
            ego_his_trajs=None,  #  dont have to because the dont use the history trajs
            ego_lcf_feat=None,  # these are not used either
        )
        # save the bev feature map
        self.prev_frame_info["prev_bev"] = outs["bev_embed"]

        bbox_list = self.model.pts_bbox_head.get_bboxes(outs, img_metas, rescale=False)

        bbox_results = []
        for i, (
            bboxes,
            scores,
            labels,
            trajs,
            map_bboxes,
            map_scores,
            map_labels,
            map_pts,
        ) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result["trajs_3d"] = trajs.cpu()
            map_bbox_result = self.model.map_pred2result(
                map_bboxes, map_scores, map_labels, map_pts
            )
            bbox_result.update(map_bbox_result)
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i].cpu()
            bbox_results.append(bbox_result)

        # note that they work with deltas
        trajectory = (
            bbox_results[0]["ego_fut_preds"][input.command].cumsum(dim=-2).numpy()
        )
        future_trajs = (
            bbox_results[0]["trajs_3d"].reshape(-1, 6, 6, 2).cumsum(dim=-2)
        )  # + bboxes.bev[:, :2].unsqueeze(1).unsqueeze(1)

        return VADInferenceOutput(
            trajectory=trajectory,
            aux_outputs=VADAuxOutputs(
                objects_in_bev=bbox_results[0]["boxes_3d"].bev.tolist(),
                object_scores=bbox_results[0]["scores_3d"].tolist(),
                object_classes=[self.classes[i] for i in bbox_results[0]["labels_3d"]],
                segmentation=None,  # bev_h, bev_w
                seg_grid_centers=None,  # bev_h, bev_w, 2 [x, y]
                future_trajs=future_trajs.tolist(),  # N x 6 modes x 6 future_timesteps x 2 (x, y)
            ),
        )


def _get_sample_input(nusc, nusc_can, scene_name, sample) -> VADInferenceInput:
    timestamp = sample["timestamp"]
    # get the cameras for this sample
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    # ego pose via lidar sensor sample data
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sample_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    ego_translation = np.array(ego_pose["translation"])
    ego_rotation_quat = Quaternion(array=ego_pose["rotation"])
    ego2global = np.eye(4)
    ego2global[:3, 3] = ego_translation
    ego2global[:3, :3] = ego_rotation_quat.rotation_matrix

    # get cameras
    camera_tokens = [sample["data"][camera_type] for camera_type in camera_types]
    # sample data for each camera
    camera_sample_data = [nusc.get("sample_data", token) for token in camera_tokens]
    # get the camera calibrations
    camera_calibrations = [
        nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        for cam in camera_sample_data
    ]
    # get the image filepaths
    image_filepaths = [
        nusc.get_sample_data(cam_token)[0] for cam_token in camera_tokens
    ]
    # get the camera instrinsics
    cam_instrinsics = [
        np.array(cam_calib["camera_intrinsic"]) for cam_calib in camera_calibrations
    ]
    # compute the camera2img and camera2ego transformations
    camera2img = []
    cam2global = []
    for i in range(len(camera_types)):
        # camera 2 image
        c2i = np.eye(4)
        c2i[:3, :3] = cam_instrinsics[i]
        camera2img.append(c2i)
        # camera 2 ego
        c2e = np.eye(4)
        c2e[:3, 3] = np.array(camera_calibrations[i]["translation"])
        c2e[:3, :3] = Quaternion(
            array=camera_calibrations[i]["rotation"]
        ).rotation_matrix
        # ego 2 global (for camera time)
        cam_e2g = nusc.get("ego_pose", camera_sample_data[i]["ego_pose_token"])
        cam_e2g_t = np.array(cam_e2g["translation"])
        cam_e2g_r = Quaternion(array=cam_e2g["rotation"])
        e2g = np.eye(4)
        e2g[:3, 3] = cam_e2g_t
        e2g[:3, :3] = cam_e2g_r.rotation_matrix
        # cam 2 global
        cam2global.append(e2g @ c2e)

    # load the images in rgb hwc format
    images = []
    for filepath in image_filepaths:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    images = np.array(images)

    # get the lidar calibration
    lidar_sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_calibration = nusc.get(
        "calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]
    )
    lidar_translation = np.array(lidar_calibration["translation"])
    lidar_rotation_quat = Quaternion(array=lidar_calibration["rotation"])
    lidar2ego = np.eye(4)
    lidar2ego[:3, 3] = lidar_translation
    lidar2ego[:3, :3] = lidar_rotation_quat.rotation_matrix

    lidar2global = ego2global @ lidar2ego
    global2lidar = np.linalg.inv(lidar2global)
    # the lidar2img should take into consideration that the the timestamps are not the same
    # because of this we go img -> cam -> ego -> global -> ego' -> lidar
    cam2lidar = [global2lidar.copy() @ c2g for c2g in cam2global]
    lidar2cam = [np.linalg.inv(c2l) for c2l in cam2lidar]
    lidar2img = [c2i @ l2c for c2i, l2c in zip(camera2img, lidar2cam)]

    # get the canbus signals
    pose_messages = nusc_can.get_messages(scene_name, "pose")
    can_times = [pose["utime"] for pose in pose_messages]
    assert np.all(np.diff(can_times) > 0), "canbus times not sorted"
    # find the pose that is less than the current timestamp and closest to it
    pose_idx = np.searchsorted(can_times, timestamp)
    # get the canbus signals in the correct order
    canbus_singal_order = ["pos", "orientation", "accel", "rotation_rate", "vel"]
    canbus_signals = np.concatenate(
        [
            np.array(pose_messages[pose_idx][signal_type])
            for signal_type in canbus_singal_order
        ]
    )

    return VADInferenceInput(
        imgs=images,
        lidar_pose=lidar2global,
        lidar2img=lidar2img,
        timestamp=timestamp,
        can_bus_signals=canbus_signals,
        command=0,  # right
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runner = VADRunner(
        config_path="/VAD/projects/configs/VAD/VAD_inference.py",
        checkpoint_path="/VAD/ckpts/VAD_base.pth",
        device=torch.device(device),
    )

    # only load this for testing
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    import matplotlib.pyplot as plt
    import cv2

    # load the first surround-cam in nusc mini
    nusc = NuScenes(version="v1.0-mini", dataroot="./data/nuscenes")
    nusc_can = NuScenesCanBus(dataroot="./data/nuscenes")
    scene_name = "scene-0103"
    scene = [s for s in nusc.scene if s["name"] == scene_name][0]
    # get the first sample in the scene
    sample = nusc.get("sample", scene["first_sample_token"])

    for i in range(60):
        inference_input = _get_sample_input(nusc, nusc_can, scene_name, sample)
        if i > 4:
            inference_input.command = 2  # straight
        plan = runner.forward_inference(inference_input)
        # plot in bev
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(inference_input.imgs[0])
        ax[0].axis("off")

        # save fig
        fig.savefig(f"{scene_name}_{str(i).zfill(3)}_{sample['timestamp']}.png")
        plt.close(fig)
        if sample["next"] == "":
            break
        sample = nusc.get("sample", sample["next"])
