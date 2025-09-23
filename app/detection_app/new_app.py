import time
import av
import cv2
import torch
import pickle
import logging
from datetime import datetime
from threading import Thread, Lock
from collections import deque
from pathlib import Path
from insightface.app import FaceAnalysis
from torch_kf import KalmanFilter, GaussianState
from app.services.detection_services import get_user_details_by_unique_id
from app.services.attendance_service import mark_attendance
import uuid
import requests
import os
from dotenv import load_dotenv
import numpy as np
from app.services.attendance_service import frame_to_base64

load_dotenv()


DEVICE = "cuda"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)


class Track:
    """Represents a tracked face."""

    name = "Unknown"

    def __init__(self, bbox, emb=None):
        self.name = Track.name
        self.kf_state = None
        self.kf = self._init_kf(bbox)
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.label = None
        self.conf = 0.0
        self.emb_smooth = deque(maxlen=10)
        if emb is not None:
            self.emb_smooth.append(emb.to(DEVICE))

        # Unknown face tracking fields
        self.temp_id = str(uuid.uuid4())  # unique ID for this track
        self.is_unknown = False  # flag if currently unknown
        self.unknown_frames = 0  # consecutive frames marked unknown
        self.last_notified_time = None  # last time we sent API notification
        self.unknown_notified = False  # whether notification was already sent
        self.first_seen = datetime.now()  # when this track first appeared
        self.frames_seen = 0  # how many frames total seen

    def _init_kf(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        F = torch.eye(7, device=DEVICE)
        F[0, 4] = 1
        F[1, 5] = 1
        F[2, 6] = 1

        H = torch.zeros(4, 7, device=DEVICE)
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1

        Q = torch.eye(7, device=DEVICE) * 0.01
        Q[4:, 4:] *= 10
        R = torch.eye(4, device=DEVICE) * 1.5

        kf = KalmanFilter(
            process_matrix=F, measurement_matrix=H, process_noise=Q, measurement_noise=R
        )

        initial_mean = torch.tensor(
            [cx, cy, w, h, 0, 0, 0], dtype=torch.float32, device=DEVICE
        )
        initial_cov = torch.eye(7, device=DEVICE) * 10
        initial_cov[4:, 4:] *= 100

        self.kf_state = GaussianState(mean=initial_mean, covariance=initial_cov)
        return kf

    def predict(self):
        self.kf_state = self.kf.predict(self.kf_state)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        measurement_mean = torch.tensor(
            [cx, cy, w, h], dtype=torch.float32, device=DEVICE
        )
        self.kf_state = self.kf.update(self.kf_state, measurement_mean)
        self.time_since_update = 0
        self.hits += 1

    def get_state(self):
        mean = self.kf_state.mean
        cx, cy, w, h = mean[:4]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [x1, y1, x2, y2]


class FrameGrabber:
    """Threaded frame grabber for RTSP streams."""

    def __init__(self, container, stream, device="cuda"):
        self.container = container
        self.stream = stream
        self.device = device
        self.frame = None
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        try:
            for packet in self.container.demux(self.stream):
                for frame in packet.decode():
                    img_np = frame.to_ndarray(format="bgr24")
                    img_tensor = torch.from_numpy(img_np).to(self.device)
                    with self.lock:
                        self.frame = img_tensor
                    if not self.running:
                        return
        except Exception as e:
            logging.error(f"FrameGrabber error: {e}")
        finally:
            self.running = False

    def read(self):
        with self.lock:
            return self.frame.clone() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=5)


class CameraStream:
    """Encapsulates full face recognition + tracking pipeline for one camera."""

    def __init__(
        self,
        camera_id,
        rtsp_url,
        company_id,
        emb_db_path="encodings/embeddings.pkl",
        checkin_cooldown=300,
        checkout_cooldown=300,
        show_window=False,
        use_gpu=True,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.emb_db_path = emb_db_path
        self.checkin_cooldown = checkin_cooldown
        self.checkout_cooldown = checkout_cooldown
        # self.show_window = show_window
        self.show_window = True
        self.use_gpu = use_gpu
        self.running = False
        self.last_seen = {}
        self.tracks = []
        self.DEVICE = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if self.use_gpu and torch.cuda.device_count() == 0:
            logging.warning("GPU requested but not available. Switching to CPU.")
            self.use_gpu = False
            self.DEVICE = "cpu"
        self.frame_grabber = None
        self.names, self.embs = self.load_embeddings(emb_db_path)
        self.app = self.init_face_app()
        self.tile_grid = (1, 1)
        self.detect_every_n_frames = 8
        self.topk_per_tile = 10
        self.conf_thresh = 0.40
        self.emb_match_thresh = 0.75
        self.cooldown = 60

        # Unknown-person handling
        self.unknown_confirm_frames = 16
        self.unknown_notify_cooldown = 30
        self.unknown_notify_url = os.getenv("THREAT_API", None)
        self.max_unknown_store = 200  
        self.company_id = company_id

    def load_embeddings(self, path):
        if not Path(path).exists():
            logging.error(f"Embedding DB not found: {path}")
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            db = pickle.load(f)
        names = [d["name"] for d in db]
        embs = torch.stack(
            [torch.tensor(d["embedding"], dtype=torch.float32) for d in db], dim=0
        ).to(self.DEVICE)
        logging.info(
            f"[{self.camera_id}] Loaded {len(names)} embeddings to {self.DEVICE}"
        )
        return names, embs

    def init_face_app(self):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if self.use_gpu else -1, det_size=(1920, 1920))
        return app

    def iou_matrix(self, tracks_bbox_list, dets_bbox_list):
        if not tracks_bbox_list or not dets_bbox_list:
            return torch.empty(0, 0, device=self.DEVICE)
        tracks = torch.stack(tracks_bbox_list).to(self.DEVICE)
        dets = torch.stack(dets_bbox_list).to(self.DEVICE)
        tl = torch.max(tracks[:, None, :2], dets[None, :, :2])
        br = torch.min(tracks[:, None, 2:], dets[None, :, 2:])
        wh = (br - tl).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area_track = ((tracks[:, 2] - tracks[:, 0]) * (tracks[:, 3] - tracks[:, 1]))[
            :, None
        ]
        area_det = ((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))[None, :]
        return inter / (area_track + area_det - inter + 1e-9)

    def associate_tracks(self, tracks, dets, iou_thresh=0.35):
        if not tracks:
            return [], [], list(range(len(dets)))
        if not dets:
            return [], list(range(len(tracks))), []
        track_boxes = [
            torch.tensor(t.get_state(), dtype=torch.float32, device=self.DEVICE)
            for t in tracks
        ]
        det_boxes = [
            torch.tensor(d["bbox"], dtype=torch.float32, device=self.DEVICE)
            for d in dets
        ]
        iou_mat = self.iou_matrix(track_boxes, det_boxes)
        matches, used_t, used_d = (
            [],
            torch.zeros(len(tracks), dtype=torch.bool, device=self.DEVICE),
            torch.zeros(len(dets), dtype=torch.bool, device=self.DEVICE),
        )
        N, M = iou_mat.shape
        flat_iou = iou_mat.flatten()
        sorted_vals, sorted_idx = torch.sort(flat_iou, descending=True)
        for val, idx in zip(sorted_vals, sorted_idx):
            if val < iou_thresh:
                break
            i = idx // M
            j = idx % M
            if used_t[i] or used_d[j]:
                continue
            matches.append((int(i), int(j)))
            used_t[i] = True
            used_d[j] = True
        u_tracks = [i for i in range(N) if not used_t[i]]
        u_dets = [j for j in range(M) if not used_d[j]]
        return matches, u_tracks, u_dets

    def split_into_tiles(self, frame):
        H, W = frame.shape[:2]
        gh, gw = self.tile_grid
        th = H // gh
        tw = W // gw
        tiles = []
        for r in range(gh):
            for c in range(gw):
                y1, y2 = r * th, (r + 1) * th if r < gh - 1 else H
                x1, x2 = c * tw, (c + 1) * tw if c < gw - 1 else W
                tiles.append(((x1, y1, x2, y2), frame[y1:y2, x1:x2]))
        return tiles

    def notify_unknown(self, track, frame):
        logging.debug("FUNCTION notify_unknown called")

        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).contiguous().cpu().numpy()
            frame = frame.astype(np.uint8)

        frame = np.asarray(frame)

        if frame.ndim == 3:
            if frame.shape[0] in (1, 3, 4):  
                # If (C, H, W) → (H, W, C)
                frame = np.transpose(frame, (1, 2, 0))
            elif frame.shape[1] in (1, 3, 4) and frame.shape[2] not in (1, 3, 4):
                frame = np.transpose(frame, (0, 2, 1))

        if frame.ndim == 2:  # grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA → BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:
            frame = frame[..., ::-1]  # RGB → BGR

        logging.error(f"[DEBUG] Normalized frame shape: {frame.shape}, dtype={frame.dtype}")

        x1, y1, x2, y2 = [
            int(round(v.item() if isinstance(v, torch.Tensor) else v))
            for v in track.get_state()
        ]

        H, W = frame.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame_copy, "Unknown", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        jpg_b64 = frame_to_base64(frame=frame_copy)

        payload = {
            "camera_id": self.camera_id,
            "company_id": self.company_id,
            "user_id": getattr(track, "temp_id", str(uuid.uuid4())),
            "file": jpg_b64,
            "type": "unknown",
        }

        try:
            resp = requests.post(self.unknown_notify_url, json=payload)
            if resp.status_code == 200:
                logging.info(f"[{self.camera_id}] Unknown notification sent for {payload['user_id']}")
            else:
                logging.warning(f"[{self.camera_id}] Notify API returned {resp.status_code}: {resp.text}")
        except Exception as e:
            logging.error(f"[{self.camera_id}] Failed to notify unknown: {e}")

    def run(self):
        logging.info(f"[{self.camera_id}] Starting camera stream pipeline")
        self.running = True
        try:
            container = av.open(
                self.rtsp_url,
                options={
                    "rtsp_transport": "tcp",
                    "stimeout": "5000000",
                    "rw_timeout": "5000000",
                    "max_delay": "500000",
                },
            )
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
        except Exception as e:
            logging.error(f"[{self.camera_id}] Failed to open RTSP stream: {e}")
            return

        self.frame_grabber = FrameGrabber(container, stream, device=self.DEVICE)
        frame_idx = 0

        while self.running:
            frame = self.frame_grabber.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1
            H, W = frame.shape[:2]

            # Predict existing tracks
            for t in self.tracks:
                t.predict()

            new_dets = []
            if frame_idx % self.detect_every_n_frames == 1:
                tiles = self.split_into_tiles(frame)
                for (x1_tile, y1_tile, x2_tile, y2_tile), crop_gpu in tiles:
                    crop_np = crop_gpu.cpu().numpy()
                    # faces = self.app.get(crop_np, max_num=self.topk_per_tile)
                    try:
                        faces = self.app.get(crop_np, max_num=self.topk_per_tile)
                    except Exception as e:
                        logging.error(f"[{self.camera_id}] Face detection error: {e}")
                        faces = []

                    for f in faces:
                        if f.det_score < self.conf_thresh:
                            continue
                        bx1, by1, bx2, by2 = f.bbox.astype(float)
                        emb_tensor = torch.tensor(
                            f.normed_embedding, dtype=torch.float32, device=self.DEVICE
                        )
                        new_dets.append(
                            {
                                "bbox": [
                                    bx1 + x1_tile,
                                    by1 + y1_tile,
                                    bx2 + x1_tile,
                                    by2 + y1_tile,
                                ],
                                "score": float(f.det_score),
                                "emb": emb_tensor,
                            }
                        )

                matches, u_tracks, u_dets = self.associate_tracks(self.tracks, new_dets)
                for track_idx, det_idx in matches:
                    track = self.tracks[track_idx]
                    det = new_dets[det_idx]
                    track.update(det["bbox"])
                    track.conf = det["score"]
                    track.emb_smooth.append(det["emb"])

                for det_idx in u_dets:
                    det = new_dets[det_idx]
                    new_track = Track(det["bbox"], det["emb"])
                    new_track.conf = det["score"]
                    self.tracks.append(new_track)
                    # Unknown bookkeeping
                    new_track.temp_id = str(uuid.uuid4())
                    new_track.is_unknown = True  # start as unknown until identified
                    new_track.unknown_frames = 1
                    new_track.frames_seen = 1
                    new_track.last_notified_time = None
                    new_track.unknown_notified = False
                    self.tracks.append(new_track)

                self.tracks = [t for t in self.tracks if t.time_since_update <= 12]

                # Face identification & unknown tracking
                for t in self.tracks:
                    # Filter out any None or empty tensors
                    valid_embs = [
                        e for e in t.emb_smooth if e is not None and e.numel() > 0
                    ]
                    if len(valid_embs) == 0:
                        continue
                    try:
                        emb = torch.mean(torch.stack(valid_embs), dim=0)
                        emb = emb / (emb.norm() + 1e-9)
                    except RuntimeError as e:
                        logging.warning(
                            f"Skipping empty embedding for track {t.temp_id}: {e}"
                        )
                        continue
                    dots = torch.matmul(self.embs, emb)
                    dists = 1.0 - dots
                    min_dist, min_idx = torch.min(dists, dim=0)
                    min_dist_value = min_dist.item()
                    k = min_idx.item()

                    now = datetime.now()

                    if min_dist_value <= self.emb_match_thresh:
                        emp_id = self.names[k]
                        t.label = emp_id
                        t.is_unknown = False
                        t.unknown_frames = 0
                        t.frames_seen += 1

                        details = get_user_details_by_unique_id(emp_id)
                        if not details:
                            logging.warning(
                                f"Employee details not found for ID: {emp_id}"
                            )
                            continue

                        t.name = details.get("name", "Unknown")

                        # Attendance logic
                        last_action_time = self.last_seen.get(emp_id)
                        if (
                            last_action_time
                            and (now - last_action_time).seconds < self.cooldown
                        ):
                            continue  # skip due to cooldown

                        # Update cooldown tracker
                        self.last_seen[emp_id] = now

                        attendance_result, error = mark_attendance(
                            emp_id,
                            camera_id=self.camera_id,
                            action="checkin",
                            frame=frame,
                        )
                        if error == "No active check-in found to check out":
                            attendance_result, error = mark_attendance(
                                emp_id, self.camera_id, action="checkin", frame=frame
                            )
                            if not error:
                                logging.info(f"{t.name} checked in at {now}")
                        elif error == "User already checked in today":
                            attendance_result, error = mark_attendance(
                                emp_id,
                                action="checkout",
                                frame=frame,
                                camera_id=self.camera_id,
                            )
                            if not error:
                                logging.info(f"{t.name} checked out at {now}")
                        elif not error:
                            logging.info(
                                f"{t.name} attendance recorded: {attendance_result}"
                            )
                        else:
                            logging.warning(f"Attendance error for {t.name}: {error}")

                    else:

                        t.is_unknown = True
                        t.frames_seen += 1
                        t.unknown_frames = getattr(t, "unknown_frames", 0) + 1

                        # Determine if we can send notification
                        can_notify = t.unknown_frames >= getattr(
                            self, "unknown_confirm_frames", 5
                        ) and (
                            t.last_notified_time is None
                            or (now - t.last_notified_time).total_seconds()
                            > getattr(self, "unknown_notify_cooldown", 60)
                        )

                        if can_notify:
                            try:
                                # notify_unknown should handle cropping, base64, bbox, camera id, timestamp
                                self.notify_unknown(t, frame)
                                t.last_notified_time = now
                                t.unknown_notified = True
                                logging.info(
                                    f"[{self.camera_id}] Unknown person notified: {t.temp_id}"
                                )
                            except Exception as e:
                                logging.exception(
                                    f"[{self.camera_id}] Failed to notify unknown: {e}"
                                )

            torch.cuda.empty_cache()
            # Visualization
            if self.show_window:
                vis = frame.cpu().numpy().copy()
                for t in self.tracks:
                    x1, y1, x2, y2 = [int(c.item()) for c in t.get_state()]
                    x1, y1, x2, y2 = (
                        max(0, x1),
                        max(0, y1),
                        min(W - 1, x2),
                        min(H - 1, y2),
                    )
                    if x2 <= x1 or y2 <= y1:
                        continue
                    tag = f"{t.name}"
                    color = (0, 255, 0) if t.label else (0, 0, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        vis,
                        tag,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                cv2.imshow(f"Camera {self.camera_id}", cv2.resize(vis, (1280, 720)))
                if cv2.waitKey(1) & 0xFF == 27:
                    logging.info(f"[{self.camera_id}] ESC pressed. Stopping camera.")
                    self.stop()
                    break

        self.cleanup()

    def stop(self):
        self.running = False

    def cleanup(self):
        if self.frame_grabber:
            self.frame_grabber.stop()
        cv2.destroyAllWindows()
        logging.info(f"[{self.camera_id}] Camera stopped and resources released.")
