import cv2
import numpy as np
import mss
import tkinter as tk
from PIL import Image, ImageTk
import torch
import kornia as K
from kornia.feature import LoFTR
import ssl
import config  # <--- 导入同目录下的配置文件

ssl._create_default_https_context = ssl._create_unverified_context


class AIMapTrackerApp:
    def __init__(self, root, minimap_region=None):
        self.root = root
        self.root.title("AI 智能雷达跟点 (双图分离)")

        self.root.attributes("-topmost", True)
        # --- 使用配置文件中的悬浮窗几何设置 ---
        self.root.geometry(config.WINDOW_GEOMETRY)

        # --- 1. 加载 AI 模型 ---
        print("正在加载 LoFTR AI 模型...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前计算设备: {self.device}")
        self.matcher = LoFTR(pretrained='outdoor').to(self.device)
        self.matcher.eval()
        print("AI 模型加载完成！")

        # --- 2. 加载【双地图】 ---
        print(f"正在加载逻辑大地图 ({config.LOGIC_MAP_PATH})...")
        self.logic_map_bgr = cv2.imread(config.LOGIC_MAP_PATH)
        if self.logic_map_bgr is None:
            raise FileNotFoundError(f"找不到逻辑地图: {config.LOGIC_MAP_PATH}！")
        self.map_height, self.map_width = self.logic_map_bgr.shape[:2]

        print(f"正在加载显示大地图 ({config.DISPLAY_MAP_PATH})...")
        self.display_map_bgr = cv2.imread(config.DISPLAY_MAP_PATH)
        if self.display_map_bgr is None:
            raise FileNotFoundError(f"找不到显示地图: {config.DISPLAY_MAP_PATH}！")

        # --- 3. 追踪状态机初始化 ---
        self.state = "GLOBAL_SCAN"  # 初始状态为全局雷达扫描
        self.last_x = 0
        self.last_y = 0

        # --- 使用配置文件中的雷达与追踪参数 ---
        self.scan_size = config.AI_SCAN_SIZE
        self.scan_step = config.AI_SCAN_STEP
        self.scan_x = 0
        self.scan_y = 0

        self.search_radius = config.AI_TRACK_RADIUS
        self.lost_frames = 0

        # 注意：AI 模式下彻底丢失几帧就切回雷达扫描。
        # 你可以考虑在 config 里单独加一个 AI_MAX_LOST_FRAMES = 3，或者直接在这里用一个较小的数字
        self.max_lost_frames = 3

        # --- 4. 截图与 UI ---
        self.sct = mss.mss()
        self.minimap_region = (
            minimap_region if minimap_region is not None else config.MINIMAP
        )

        # --- 使用配置文件中的视野大小 (VIEW_SIZE) ---
        self.canvas = tk.Canvas(root, width=config.VIEW_SIZE, height=config.VIEW_SIZE, bg='#2b2b2b')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_on_canvas = None
        self._ui_photo_ref = None

        self.update_tracker()

    def preprocess_image(self, img_bgr):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        new_h = h - (h % 8)
        new_w = w - (w % 8)
        img_gray = cv2.resize(img_gray, (new_w, new_h))
        tensor = K.image_to_tensor(img_gray, False).float() / 255.0
        return tensor.to(self.device)

    def update_tracker(self):
        # 1. 获取小地图
        screenshot = self.sct.grab(self.minimap_region)
        minimap_bgr = np.array(screenshot)[:, :, :3]

        found = False
        display_crop = None
        half_view = config.VIEW_SIZE // 2  # 视野的一半，用于计算裁剪范围

        # ==========================================
        # 状态机：确定当前的搜索区域
        # ==========================================
        if self.state == "GLOBAL_SCAN":
            x1 = self.scan_x
            y1 = self.scan_y
            x2 = min(self.map_width, x1 + self.scan_size)
            y2 = min(self.map_height, y1 + self.scan_size)

            display_crop = self.display_map_bgr[y1:y2, x1:x2].copy()
            display_crop = cv2.resize(display_crop, (config.VIEW_SIZE, int(config.VIEW_SIZE * (y2 - y1) / (x2 - x1))))
            cv2.putText(display_crop, f"Global Scan: X:{x1} Y:{y1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

        else:  # TRACKING_LOCAL
            x1 = max(0, self.last_x - self.search_radius)
            y1 = max(0, self.last_y - self.search_radius)
            x2 = min(self.map_width, self.last_x + self.search_radius)
            y2 = min(self.map_height, self.last_y + self.search_radius)

        # 2. 从【逻辑地图】上截取搜索区域，喂给 AI
        local_logic_map = self.logic_map_bgr[y1:y2, x1:x2]

        if local_logic_map.shape[0] >= 16 and local_logic_map.shape[1] >= 16:
            tensor_mini = self.preprocess_image(minimap_bgr)
            tensor_big_local = self.preprocess_image(local_logic_map)

            input_dict = {"image0": tensor_mini, "image1": tensor_big_local}

            with torch.no_grad():
                correspondences = self.matcher(input_dict)

            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()

            # --- 使用配置文件中的置信度阈值 ---
            valid_idx = confidence > config.AI_CONFIDENCE_THRESHOLD
            mkpts0 = mkpts0[valid_idx]
            mkpts1 = mkpts1[valid_idx]

            # ==========================================
            # AI 结果处理与状态切换
            # ==========================================
            # --- 使用配置文件中的最小匹配点数 ---
            if len(mkpts0) >= config.AI_MIN_MATCH_COUNT:
                # --- 使用配置文件中的 RANSAC 误差阈值 ---
                M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, config.AI_RANSAC_THRESHOLD)

                if M is not None:
                    h, w = minimap_bgr.shape[:2]
                    center_pt = np.float32([[[w / 2, h / 2]]])
                    dst_center_local = cv2.perspectiveTransform(center_pt, M)

                    center_x = int(dst_center_local[0][0][0] + x1)
                    center_y = int(dst_center_local[0][0][1] + y1)

                    if 0 <= center_x < self.map_width and 0 <= center_y < self.map_height:
                        found = True

                        self.last_x = center_x
                        self.last_y = center_y
                        self.state = "LOCAL_TRACK"
                        self.lost_frames = 0

                        # 从【显示地图】截取周围的视野画出来 (使用 config.VIEW_SIZE)
                        vy1 = max(0, center_y - half_view)
                        vy2 = min(self.map_height, center_y + half_view)
                        vx1 = max(0, center_x - half_view)
                        vx2 = min(self.map_width, center_x + half_view)

                        display_crop = self.display_map_bgr[vy1:vy2, vx1:vx2].copy()

                        local_cx = center_x - vx1
                        local_cy = center_y - vy1
                        cv2.circle(display_crop, (local_cx, local_cy), radius=10, color=(0, 0, 255), thickness=-1)
                        cv2.circle(display_crop, (local_cx, local_cy), radius=12, color=(255, 255, 255), thickness=2)

        # ==========================================
        # 丢失处理与雷达网格更新
        # ==========================================
        if not found:
            if self.state == "LOCAL_TRACK":
                self.lost_frames += 1
                if self.lost_frames <= self.max_lost_frames:
                    vy1 = max(0, self.last_y - half_view)
                    vy2 = min(self.map_height, self.last_y + half_view)
                    vx1 = max(0, self.last_x - half_view)
                    vx2 = min(self.map_width, self.last_x + half_view)
                    display_crop = self.display_map_bgr[vy1:vy2, vx1:vx2].copy()

                    local_cx = self.last_x - vx1
                    local_cy = self.last_y - vy1
                    cv2.circle(display_crop, (local_cx, local_cy), radius=10, color=(0, 255, 255), thickness=-1)
                else:
                    print("彻底丢失目标，启动全局雷达扫描...")
                    self.state = "GLOBAL_SCAN"
                    self.scan_x = 0
                    self.scan_y = 0
                    display_crop = np.zeros((config.VIEW_SIZE, config.VIEW_SIZE, 3), dtype=np.uint8)
                    cv2.putText(display_crop, "Radar Initializing...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)

            elif self.state == "GLOBAL_SCAN":
                self.scan_x += self.scan_step
                if self.scan_x >= self.map_width:
                    self.scan_x = 0
                    self.scan_y += self.scan_step
                    if self.scan_y >= self.map_height:
                        self.scan_x = 0
                        self.scan_y = 0

        # ==========================================
        # 统一渲染输出到 UI
        # ==========================================
        display_rgb = cv2.cvtColor(display_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_rgb)

        final_img = Image.new('RGB', (config.VIEW_SIZE, config.VIEW_SIZE), (43, 43, 43))
        # 将画面居中粘贴
        final_img.paste(pil_image,
                        (max(0, half_view - pil_image.width // 2), max(0, half_view - pil_image.height // 2)))

        ref = self._ui_photo_ref
        if ref is None:
            self._ui_photo_ref = ImageTk.PhotoImage(final_img)
        else:
            ref.paste(final_img)
        self.tk_image = self._ui_photo_ref

        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)

        # --- 使用配置文件中的刷新频率 ---
        self.root.after(config.AI_REFRESH_RATE, self.update_tracker)


if __name__ == "__main__":
    from screen_pick import run_with_screen_pick

    run_with_screen_pick(
        AIMapTrackerApp,
        title_hint="AI 智能雷达跟点 (双图分离)",
    )