import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3CrowdCounter:
    def __init__(
        self,
        record_root="/home/zys/city_isaacgym/record",
        camera_intrinsic=None,
        min_distance=0.8,
        device="cuda"
    ):
        self.record_root = record_root
        self.json_path = os.path.join(record_root, "dataset_index.json")
        self.min_distance = min_distance
        self.device = device

        self.camera_intrinsic = camera_intrinsic or {
            "fx": 200.0,
            "fy": 200.0,
            "cx": 128.0,
            "cy": 128.0
        }

        self.sam3_model = build_sam3_image_model().to(device)
        self.sam3_processor = Sam3Processor(self.sam3_model)

        self.data = []
        self.world_points = []

    # ============================================================
    def read_json(self):
        with open(self.json_path, "r") as f:
            self.data = json.load(f)
        print(f"[OK] Loaded {len(self.data)} frames")

    def foot_points_from_masks(self, rgb_path, depth_path, text_prompt="person"):
        """
        直接使用 SAM3 masks 和 depth.npy 找每个人脚底坐标
        """
        # 读取 RGB 和 depth
        image = Image.open(rgb_path).convert("RGB")
        depth = np.load(depth_path, allow_pickle=True)  # [H, W]

        # SAM3 推理
        state = self.sam3_processor.set_image(image)
        output = self.sam3_processor.set_text_prompt(state=state, prompt=text_prompt)
        masks = output["masks"]  # [N, 1, H, W]

        # 转 numpy
        if "torch" in str(type(masks)):
            masks = masks.cpu().numpy()

        foot_points = []
        N = masks.shape[0]

        for i in range(N):
            mask = masks[i, 0]  # [H, W]
            mask_bool = mask > 0.5
            if np.any(mask_bool):
                ys, xs = np.where(mask_bool)
                # 脚底 = depth 最小点（离相机最近）
                depths = depth[ys, xs]
                idx = np.argmin(depths)
                x = xs[idx]
                y = ys[idx]
                z = depth[y, x]
                foot_points.append((x, y, z))
            else:
                foot_points.append(None)

        return foot_points

    # ------------------------------------------------------------
    def draw_feet_on_rgb(self, rgb_path, feet, save_path=None):
        """
        在 RGB 图上绘制脚底点
        """
        if save_path is None:
            base = os.path.basename(rgb_path).replace("_rgb.jpg", "")
            save_path = os.path.join(self.record_root, f"{base}_feet.png")

        img = cv2.imread(rgb_path)
        for i, point in enumerate(feet):
            if point is None:
                continue
            x, y, _ = point
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(img, f"{i+1}", (int(x)+4, int(y)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        cv2.imwrite(save_path, img)
        return save_path

    # ------------------------------------------------------------
    def image_to_world(self, points, pose):
        fx, fy = self.camera_intrinsic["fx"], self.camera_intrinsic["fy"]
        cx, cy = self.camera_intrinsic["cx"], self.camera_intrinsic["cy"]

        cam_pos = np.array(pose[:3])
        roll, pitch, yaw = pose[3:]

        Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        R = Rz @ Ry @ Rx

        pts = []
        for x, y, z in points:
            X = (x - cx) / fx * z
            Y = (y - cy) / fy * z
            pts.append(R @ np.array([X, Y, z]) + cam_pos)

        return np.array(pts)

    # ------------------------------------------------------------
    # def remove_close_points(self, points):
    #     if len(points) == 0:
    #         return points
    #     d = cdist(points, points)
    #     keep = np.ones(len(points), bool)
    #     for i in range(len(points)):
    #         for j in range(i+1, len(points)):
    #             if d[i, j] < self.min_distance:
    #                 keep[j] = False
    #     return points[keep]


    def remove_close_points(self, points):
        """
        对所有点去重，保留距离 >= min_distance 的点。
        使用 KDTree 进行快速邻域搜索。
        """
        from scipy.spatial import KDTree

        if len(points) == 0:
            return points

        points = np.array(points)
        tree = KDTree(points)
        N = len(points)
        keep_mask = np.ones(N, dtype=bool)

        for i in range(N):
            if not keep_mask[i]:
                continue
            # 查询 i 点附近所有小于 min_distance 的点
            idxs = tree.query_ball_point(points[i], r=self.min_distance)
            # 除了自己，其他都标记为 False
            for j in idxs:
                if j != i:
                    keep_mask[j] = False

        return points[keep_mask]

    def count_total(self, text_prompt="person"):
        all_pts = []
        for frame in self.data:
            rgb_path = os.path.join(self.record_root, frame["rgb_path"])
            depth_path = os.path.join(self.record_root, frame["depth_npy_path"])

            # 获取脚底点
            feet = self.foot_points_from_masks(rgb_path, depth_path, text_prompt)

            # 打印该图片人数（非 None 的脚底点数量）
            num_people = sum([1 for p in feet if p is not None])
            print(f"{frame['rgb_path']} detected {num_people} people")

            # 绘制到 RGB
            self.draw_feet_on_rgb(rgb_path, feet)
            # 转到世界坐标
            w = self.image_to_world([p for p in feet if p is not None], frame["pose"])
            if len(w) > 0:
                all_pts.append(w)

        if not all_pts:
            return 0

        all_pts = np.vstack(all_pts)
        all_pts = self.remove_close_points(all_pts)
        return len(all_pts)



if __name__ == "__main__":
    counter = SAM3CrowdCounter("/home/zys/city_isaacgym/record")
    counter.read_json()
    total = counter.count_total("person")
    print("Total people:", total)

    GT = 226
    mae =abs(GT - total)
    print("MAE:", mae)
    mape = mae / GT * 100
    print(f"MAPE = {mape:.2f}%")



# import os

# DATA_ROOT = "/home/zys/city_isaacgym/data_offline"


# def compute_ground_truth_for_all():
#     """
#     遍历：
#     data_offline/
#         env_x/
#             num_points/
#                 height/
#                     -> 计算总人数
#                     -> 保存 GroundTruth.txt
#     """

#     # 第一层：env_index
#     for env_name in sorted(os.listdir(DATA_ROOT)):
#         env_path = os.path.join(DATA_ROOT, env_name)
#         if not os.path.isdir(env_path):
#             continue

#         print(f"\n[ENV] {env_name}")

#         # 第二层：num_points
#         for num_points in sorted(os.listdir(env_path)):
#             num_path = os.path.join(env_path, num_points)
#             if not os.path.isdir(num_path):
#                 continue

#             print(f"  [NUM_POINTS] {num_points}")

#             # 第三层：height
#             for height in sorted(os.listdir(num_path)):
#                 height_path = os.path.join(num_path, height)
#                 if not os.path.isdir(height_path):
#                     continue

#                 print(f"    [HEIGHT] {height}")

#                 try:
#                     # -----------------------------
#                     # 使用 SAM3 统计
#                     # -----------------------------
#                     counter = SAM3CrowdCounter(height_path)
#                     counter.read_json()
#                     total = counter.count_total("person")

#                     # -----------------------------
#                     # 保存 GroundTruth.txt
#                     # -----------------------------
#                     save_path = os.path.join(height_path, "GroundTruth.txt")
#                     with open(save_path, "w") as f:
#                         f.write(str(total))

#                     print(f"      ✔ Total people = {total}")

#                 except Exception as e:
#                     print(f"      ✘ Failed: {e}")


# if __name__ == "__main__":
#     compute_ground_truth_for_all()
