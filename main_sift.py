"""
SIFT 跟点：仅在逻辑图有效掩膜内提特征；锚点可缓存。
大地图锚点数量由 SIFT_MAP_NFEATURES 控制：0 表示不限制（全量锚点，精度优先）。
有上一帧位置时仅在「空间邻域」内筛选锚点做 FLANN，不截断特征总数。
"""
from __future__ import annotations

import math
import threading
import time
import cv2
import mss
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

import config

try:
    cv2.setUseOptimized(True)
except Exception:
    pass
from map_mask import (
    load_logic_bgr_and_region_mask,
    try_load_sift_anchors,
    save_sift_anchors,
)


def _downscale_gray_max_edge(gray: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return gray
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return gray
    s = max_edge / m
    nw, nh = int(round(w * s)), int(round(h * s))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def _minimap_inscribed_ellipse_mask(h: int, w: int, axis_scale: float) -> np.ndarray:
    """与截屏矩形同尺寸，内接椭圆内为 255，四角为 0。方形截屏时为圆。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    if h < 2 or w < 2:
        return mask
    sc = max(0.05, min(1.0, float(axis_scale)))
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    ax = max(w * 0.5 * sc, 1.0)
    ay = max(h * 0.5 * sc, 1.0)
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (int(round(ax)), int(round(ay))),
        0,
        0,
        360,
        255,
        -1,
    )
    return mask


def _create_sift_map():
    """大地图：0 = 不限制 nfeatures。"""
    nf = int(getattr(config, "SIFT_MAP_NFEATURES", 0) or 0)
    if nf <= 0:
        return cv2.SIFT_create()
    return cv2.SIFT_create(nfeatures=nf)


def _create_sift_query():
    nq = int(getattr(config, "SIFT_QUERY_NFEATURES", 500) or 0)
    if nq <= 0:
        return cv2.SIFT_create()
    return cv2.SIFT_create(nfeatures=max(64, nq))


class SiftMapTrackerApp:
    def __init__(self, root, minimap_region=None):
        self.root = root
        self.root.title("SIFT 双地图跟点 (逻辑与显示分离)")

        self.root.attributes("-topmost", True)
        self.root.geometry(config.WINDOW_GEOMETRY)

        self.last_x = None
        self.last_y = None
        self.lost_frames = 0
        self.MAX_LOST_FRAMES = config.MAX_LOST_FRAMES

        print(f"正在加载逻辑大地图 ({config.LOGIC_MAP_PATH})…")
        self.logic_map_bgr, self._region_mask = load_logic_bgr_and_region_mask(
            config
        )
        self.map_height, self.map_width = self.logic_map_bgr.shape[:2]
        valid_px = int(np.count_nonzero(self._region_mask))
        print(
            f"有效地图区域像素: {valid_px} / {self.map_width * self.map_height} "
            f"（仅在有效区内提 SIFT）"
        )

        logic_map_gray = cv2.cvtColor(self.logic_map_bgr, cv2.COLOR_BGR2GRAY)

        print(f"正在加载显示大地图 ({config.DISPLAY_MAP_PATH})…")
        self.display_map_bgr = cv2.imread(config.DISPLAY_MAP_PATH)
        if self.display_map_bgr is None:
            raise FileNotFoundError(
                f"找不到显示地图文件: {config.DISPLAY_MAP_PATH}，请检查路径！"
            )

        dh, dw = self.display_map_bgr.shape[:2]
        if dh != self.map_height or dw != self.map_width:
            raise ValueError(
                f"严重错误：逻辑地图({self.map_width}x{self.map_height}) 与 显示地图({dw}x{dh}) 尺寸不一致！"
            )

        self.clahe = cv2.createCLAHE(
            clipLimit=config.SIFT_CLAHE_LIMIT, tileGridSize=(8, 8)
        )
        print("正在对逻辑地图进行 CLAHE…")
        logic_map_gray = self.clahe.apply(logic_map_gray)

        kp_big, des_big = try_load_sift_anchors(config, self.map_height, self.map_width)
        if des_big is None or kp_big is None:
            print("正在提取逻辑地图 SIFT（仅有效区域，首次较慢）…")
            sift_map = _create_sift_map()
            kp_big, des_big = sift_map.detectAndCompute(
                logic_map_gray, self._region_mask
            )
            if des_big is None or len(kp_big) == 0:
                raise RuntimeError(
                    "有效区域内未得到 SIFT 描述子：请检查掩膜/alpha 是否过严，或略放宽区域。"
                )
            print(f"✅ 锚点数量: {len(kp_big)}（已按有效区域；大地图未做数量截断）")
            save_sift_anchors(config, kp_big, des_big, self.map_height, self.map_width)
        else:
            print(f"✅ 使用缓存锚点: {len(kp_big)} 个")

        self.kp_big = kp_big
        self.des_big = des_big
        self._kp_xy = np.array(
            [[kp.pt[0], kp.pt[1]] for kp in self.kp_big], dtype=np.float32
        )

        self.sift = _create_sift_query()

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=int(getattr(config, "SIFT_FLANN_CHECKS", 28)))
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self._rng = np.random.default_rng()

        # mss 在 Win 上依赖线程局部 DC；须在调用 grab 的同一线程内创建并复用 MSS 实例
        self._mss_tls = threading.local()
        self.minimap_region = (
            minimap_region if minimap_region is not None else config.MINIMAP
        )

        self.canvas = tk.Canvas(
            root, width=config.VIEW_SIZE, height=config.VIEW_SIZE, bg="#2b2b2b"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_on_canvas = None
        self._track_async_busy = False
        self._fps_last_t: float | None = None
        self._fps_ema = 0.0
        self._track_inertial = False
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None
        self._smooth_tick_last_t: float | None = None

        self.update_tracker()
        _di = int(getattr(config, "SIFT_DISPLAY_INTERP_MS", 0) or 0)
        if _di > 0:
            self.root.after(_di, self._smooth_display_tick)

    def _get_mss_sct(self):
        """每个线程各自持有一个 mss.mss()，禁止跨线程复用。"""
        sct = getattr(self._mss_tls, "sct", None)
        if sct is None:
            self._mss_tls.sct = mss.mss()
            sct = self._mss_tls.sct
        return sct

    def _force_fullmap_match_state(self, lx, ly, lf: int) -> bool:
        ff = int(getattr(config, "SIFT_FORCE_FULLMAP_LOST_FRAMES", 10))
        if lx is None or ly is None:
            return True
        return lf >= ff

    def _select_train_kp_des_state(self, lx, ly, lf: int):
        if self._force_fullmap_match_state(lx, ly, lf):
            return self.kp_big, self.des_big
        r = float(getattr(config, "SIFT_LOCAL_SEARCH_RADIUS", 720))
        min_a = int(getattr(config, "SIFT_LOCAL_MIN_ANCHORS", 500))
        if r <= 0:
            return self.kp_big, self.des_big
        cx, cy = float(lx), float(ly)
        xy = self._kp_xy
        m = (
            (xy[:, 0] >= cx - r)
            & (xy[:, 0] <= cx + r)
            & (xy[:, 1] >= cy - r)
            & (xy[:, 1] <= cy + r)
        )
        idx = np.flatnonzero(m)
        if idx.size < min_a:
            return self.kp_big, self.des_big
        cap = int(getattr(config, "SIFT_LOCAL_MAX_ANCHORS", 0) or 0)
        if cap > 0 and idx.size > cap:
            idx = self._rng.choice(idx, size=cap, replace=False)
            idx.sort()
        return [self.kp_big[i] for i in idx], self.des_big[idx]

    def _compose_display_view(
        self,
        center_x: float | None,
        center_y: float | None,
        is_inertial: bool,
    ) -> np.ndarray:
        """按大地图显示层裁剪窗口并画位置点；center 可为亚像素（插帧平滑）。"""
        vs = int(config.VIEW_SIZE)
        half = int(getattr(config, "VIEW_MAP_HALF_SIZE", vs // 2))
        half = max(half, vs // 2)

        if center_x is None or center_y is None:
            display_crop = np.zeros((vs, vs, 3), dtype=np.uint8)
            cv2.putText(
                display_crop,
                "SIFT Searching...",
                (70, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            return display_crop

        cx = float(center_x)
        cy = float(center_y)
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        y1 = max(0, cy_i - half)
        y2 = min(self.map_height, cy_i + half)
        x1 = max(0, cx_i - half)
        x2 = min(self.map_width, cx_i + half)
        roi = self.display_map_bgr[y1:y2, x1:x2]
        ch, cw = roi.shape[:2]
        loc_x = cx - float(x1)
        loc_y = cy - float(y1)
        if cw >= 1 and ch >= 1:
            display_crop = cv2.resize(
                roi, (vs, vs), interpolation=cv2.INTER_AREA
            )
            lx_s = int(round(loc_x * vs / cw))
            ly_s = int(round(loc_y * vs / ch))
            lx_s = max(0, min(vs - 1, lx_s))
            ly_s = max(0, min(vs - 1, ly_s))
            r_in = max(4, int(round(10 * vs / max(cw, ch))))
            r_out = max(5, int(round(12 * vs / max(cw, ch))))
            if not is_inertial:
                cv2.circle(
                    display_crop,
                    (lx_s, ly_s),
                    radius=r_in,
                    color=(0, 0, 255),
                    thickness=-1,
                )
                cv2.circle(
                    display_crop,
                    (lx_s, ly_s),
                    radius=r_out,
                    color=(255, 255, 255),
                    thickness=2,
                )
            else:
                cv2.circle(
                    display_crop,
                    (lx_s, ly_s),
                    radius=r_in,
                    color=(0, 255, 255),
                    thickness=-1,
                )
                cv2.circle(
                    display_crop,
                    (lx_s, ly_s),
                    radius=r_out,
                    color=(0, 150, 150),
                    thickness=2,
                )
        else:
            display_crop = np.zeros((vs, vs, 3), dtype=np.uint8)
        return display_crop

    def _run_tracking_core(
        self,
    ) -> tuple[int | None, int | None, int, bool]:
        """
        截屏 + SIFT + 匹配。可在后台线程调用（勿碰 Tk）。
        返回 (last_x, last_y, lost_frames, is_inertial)；显示由 _compose_display_view / 插帧 负责。
        """
        lx, ly, lf = self.last_x, self.last_y, self.lost_frames

        clear_after = int(getattr(config, "SIFT_CLEAR_LOCK_AFTER_LOST_FRAMES", 0))
        if clear_after > 0 and lf >= clear_after:
            lx, ly, lf = None, None, 0

        screenshot = self._get_mss_sct().grab(self.minimap_region)
        minimap_bgr = np.array(screenshot)[:, :, :3]
        minimap_gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        minimap_gray = self.clahe.apply(minimap_gray)

        qe = int(getattr(config, "SIFT_QUERY_MAX_EDGE", 256))
        mini_gray = _downscale_gray_max_edge(minimap_gray, qe)
        mh, mw = mini_gray.shape[:2]
        if getattr(config, "SIFT_MINIMAP_USE_INSCRIBED_ELLIPSE", True):
            esc = float(getattr(config, "SIFT_MINIMAP_ELLIPSE_SCALE", 1.0))
            mask_mini = _minimap_inscribed_ellipse_mask(mh, mw, esc)
        else:
            mask_mini = None
        kp_mini, des_mini = self.sift.detectAndCompute(mini_gray, mask_mini)

        min_kp = int(getattr(config, "SIFT_MINIMAP_MIN_KP", 8))
        if des_mini is not None and len(kp_mini) < min_kp:
            des_mini = None

        kp_train, des_train = self._select_train_kp_des_state(lx, ly, lf)

        found = False
        center_x, center_y = None, None
        is_inertial = False

        if des_mini is not None and len(kp_mini) >= 2 and des_train is not None:
            nt = len(des_train)
            use_bf = nt < int(getattr(config, "SIFT_USE_BF_BELOW", 3500))
            if use_bf:
                matches = self.bf.knnMatch(des_mini, des_train, k=2)
            else:
                matches = self.flann.knnMatch(des_mini, des_train, k=2)

            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < config.SIFT_MATCH_RATIO * n.distance:
                        good_matches.append(m)

            min_need = int(config.SIFT_MIN_MATCH_COUNT)
            if self._force_fullmap_match_state(lx, ly, lf):
                min_need += int(getattr(config, "SIFT_RELOC_EXTRA_MIN_MATCH", 0))

            if len(good_matches) >= min_need:
                src_pts = np.float32(
                    [kp_mini[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp_train[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, config.SIFT_RANSAC_THRESHOLD
                )

                if M is not None:
                    h_m, w_m = mini_gray.shape[:2]
                    center_pt = np.float32([[[w_m / 2.0, h_m / 2.0]]])
                    dst_center = cv2.perspectiveTransform(center_pt, M)
                    temp_x = float(dst_center[0][0][0])
                    temp_y = float(dst_center[0][0][1])

                    if 0 <= temp_x < self.map_width and 0 <= temp_y < self.map_height:
                        found = True
                        center_x = int(round(temp_x))
                        center_y = int(round(temp_y))
                        lx, ly = center_x, center_y
                        lf = 0

        if not found and lx is not None and ly is not None:
            lf += 1
            if lf <= self.MAX_LOST_FRAMES:
                found = True
                center_x, center_y = lx, ly
                is_inertial = True

        return lx, ly, lf, is_inertial

    def _bump_track_fps_ema(self) -> None:
        """仅在每次跟踪完成时调用，表示「定位更新率」而非界面重绘率。"""
        now = time.perf_counter()
        if self._fps_last_t is not None:
            dt = now - self._fps_last_t
            if dt > 1e-9:
                inst = 1.0 / dt
                if self._fps_ema <= 1e-6:
                    self._fps_ema = inst
                else:
                    self._fps_ema = 0.88 * self._fps_ema + 0.12 * inst
        self._fps_last_t = now

    def _overlay_fps_on(self, display_bgr: np.ndarray) -> np.ndarray:
        if not getattr(config, "SIFT_SHOW_FPS", True):
            return display_bgr

        _, w = display_bgr.shape[:2]
        label = f"FPS {self._fps_ema:.1f}" if self._fps_ema > 0.15 else "FPS ---"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.72, w / 520.0))
        thickness = max(1, int(round(scale * 2)))
        (tw, th), _bl = cv2.getTextSize(label, font, scale, thickness)
        pad = 6
        org_x = w - tw - pad
        org_y = pad + th
        cv2.putText(
            display_bgr,
            label,
            (org_x, org_y),
            font,
            scale,
            (160, 255, 200),
            thickness,
            cv2.LINE_AA,
        )
        return display_bgr

    def _apply_tracker_ui(self, display_bgr: np.ndarray) -> None:
        display_bgr = self._overlay_fps_on(display_bgr)
        display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(display_rgb))
        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(
                0, 0, anchor=tk.NW, image=self.tk_image
            )
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)

    def _snap_smooth_on_teleport(self, lx: int, ly: int) -> None:
        tp = float(getattr(config, "SIFT_DISPLAY_TELEPORT_PX", 200.0))
        if self._smooth_x is None or self._smooth_y is None:
            self._smooth_x, self._smooth_y = float(lx), float(ly)
            return
        if math.hypot(float(lx) - self._smooth_x, float(ly) - self._smooth_y) > tp:
            self._smooth_x, self._smooth_y = float(lx), float(ly)

    def _on_tracker_result(
        self, lx: int | None, ly: int | None, lf: int, is_inertial: bool
    ) -> None:
        self.last_x, self.last_y, self.lost_frames = lx, ly, lf
        self._track_inertial = is_inertial
        if lx is None or ly is None:
            self._smooth_x = None
            self._smooth_y = None
        else:
            self._snap_smooth_on_teleport(lx, ly)

        self._bump_track_fps_ema()
        interp_ms = int(getattr(config, "SIFT_DISPLAY_INTERP_MS", 0) or 0)
        if interp_ms <= 0:
            img = self._compose_display_view(
                float(lx) if lx is not None else None,
                float(ly) if ly is not None else None,
                is_inertial,
            )
            self._apply_tracker_ui(img)

    def _smooth_display_tick(self) -> None:
        interp_ms = int(getattr(config, "SIFT_DISPLAY_INTERP_MS", 0) or 0)
        if interp_ms <= 0:
            return

        lx, ly = self.last_x, self.last_y
        tau = float(getattr(config, "SIFT_DISPLAY_SMOOTH_TAU", 0.11))
        now = time.perf_counter()
        if self._smooth_tick_last_t is None:
            dt = interp_ms * 0.001
        else:
            dt = max(1e-4, now - self._smooth_tick_last_t)
        self._smooth_tick_last_t = now

        if lx is not None and ly is not None:
            ax, ay = float(lx), float(ly)
            if self._smooth_x is None or self._smooth_y is None:
                self._smooth_x, self._smooth_y = ax, ay
            else:
                k = 1.0 - math.exp(-dt / tau) if tau > 1e-6 else 1.0
                self._smooth_x += k * (ax - self._smooth_x)
                self._smooth_y += k * (ay - self._smooth_y)

            img = self._compose_display_view(
                self._smooth_x, self._smooth_y, self._track_inertial
            )
            self._apply_tracker_ui(img)
        else:
            self._smooth_x = None
            self._smooth_y = None
            img = self._compose_display_view(None, None, False)
            self._apply_tracker_ui(img)

        self.root.after(interp_ms, self._smooth_display_tick)

    def _tracker_finish_async(self, pack: tuple) -> None:
        self._track_async_busy = False
        lx, ly, lf, is_inertial = pack
        self._on_tracker_result(lx, ly, lf, is_inertial)
        self.root.after(config.SIFT_REFRESH_RATE, self.update_tracker)

    def _tracker_async_error(self, err: BaseException) -> None:
        self._track_async_busy = False
        print(f"跟踪线程异常: {err}")
        self.root.after(config.SIFT_REFRESH_RATE, self.update_tracker)

    def update_tracker(self) -> None:
        use_bg = getattr(config, "SIFT_TRACK_IN_BACKGROUND", True)
        if use_bg:
            if self._track_async_busy:
                self.root.after(config.SIFT_REFRESH_RATE, self.update_tracker)
                return
            self._track_async_busy = True

            def worker() -> None:
                try:
                    pack = self._run_tracking_core()
                    self.root.after(0, lambda p=pack: self._tracker_finish_async(p))
                except Exception as e:
                    self.root.after(0, lambda ex=e: self._tracker_async_error(ex))

            threading.Thread(target=worker, daemon=True).start()
        else:
            pack = self._run_tracking_core()
            lx, ly, lf, is_inertial = pack
            self._on_tracker_result(lx, ly, lf, is_inertial)
            self.root.after(config.SIFT_REFRESH_RATE, self.update_tracker)


if __name__ == "__main__":
    from screen_pick import run_with_screen_pick

    run_with_screen_pick(
        SiftMapTrackerApp,
        title_hint="SIFT 双地图跟点 (逻辑与显示分离)",
    )
