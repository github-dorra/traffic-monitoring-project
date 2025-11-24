from collections import defaultdict
from time import time
import cv2
import numpy as np
from ultralytics.utils.checks import check_imshow

class SpeedEstimator:
    def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2, spdl_dist_thresh=10):
        self.reg_pts = reg_pts if reg_pts is not None else [(0, 288), (1019, 288)]
        self.names = names
        self.trk_history = defaultdict(list)  # dic stoke historiques de positions 
        self.view_img = view_img
        self.tf = line_thickness
        self.spd = {}                         # stocke vitesse actuel en km/h
        self.trkd_ids = []
        self.spdl = spdl_dist_thresh
        self.trk_pt = {}
        self.trk_pp = {}
        self.speed_limit = 60                 # seuil vitesse km/h
        self.env_check = check_imshow(warn=True)
        self.speeding_count = 0
        self.counted_ids = set()
        

        # Panneau info
        self.panel_width = 350
        self.panel_margin = 15
        self.panel_height = 250

        # Comptage de véhicules
        self.total_count = 0
        self.count_by_type = {"car": 0, "truck": 0, "bus": 0 , "train":0 , "bicycle":0 ,"motorcycle":0 }
        self.speed_history = []   #liste des vitesses moyennes 
        
        # variables de congestion
        self.jam_detected = False
        self.jam_threshold_density = 10      # nb véhicules pour congestion
        self.jam_speed_threshold = 5         # km/h
        self.jam_distance_threshold = 20     # pixels   = 1 metre 
        self.jam_window = 30                 # nb frames pour analyse
        self.seen_ids = set()

        # Conversion pixels -> mètres (à adapter selon ta vidéo)
        self.pixels_to_meters = 0.05  # 1 pixel = 5 cm par exemple

    # ------------------- Distance minimale -------------------
    def compute_min_vehicle_distance(self, boxes):
        min_dist = 9999
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cx1 = (x1 + x2) / 2
            cy1 = (y1 + y2) / 2
            for j in range(i + 1, len(boxes)):
                x1b, y1b, x2b, y2b = boxes[j]
                cx2 = (x1b + x2b) / 2
                cy2 = (y1b + y2b) / 2
                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                min_dist = min(min_dist, dist)
        return min_dist

    # ------------------- Détection embouteillage -------------------
    def detect_traffic_jam(self, boxes):
        density = self.total_count >= self.jam_threshold_density

        if len(self.speed_history) < self.jam_window // 2:
            self.jam_detected = False
            return False

        avg_speed = sum(self.speed_history) / len(self.speed_history)
        low_speed = avg_speed <= self.jam_speed_threshold

        if len(self.speed_history) > 5:
            variance = max(self.speed_history) - min(self.speed_history)
        else:
            variance = 999
        stable = variance < 3

        min_dist = self.compute_min_vehicle_distance(boxes)
        close_vehicles = min_dist <= self.jam_distance_threshold

        self.jam_detected = density and low_speed and stable and close_vehicles
        return self.jam_detected

    # ------------------- Lignes et panneaux -------------------
    def draw_dashed_line(self, img, pt1, pt2, color=(0, 0, 0), thickness=2, dash_len=15, space_len=10):
        dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
        direction = ((pt2[0] - pt1[0]) / dist, (pt2[1] - pt1[1]) / dist)
        for i in range(0, dist, dash_len + space_len):
            start = (int(pt1[0] + direction[0] * i), int(pt1[1] + direction[1] * i))
            end = (int(pt1[0] + direction[0] * min(i + dash_len, dist)),
                   int(pt1[1] + direction[1] * min(i + dash_len, dist)))
            cv2.line(img, start, end, color, thickness, lineType=cv2.LINE_AA)

    def draw_speed_panel(self, img):
        panel_x = img.shape[1] - self.panel_width - self.panel_margin
        panel_y = self.panel_margin
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + self.panel_height),
                      (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (panel_x, panel_y), (panel_x + self.panel_width, panel_y + self.panel_height),
                      (0, 0, 200), 2)

        font_title = cv2.FONT_HERSHEY_SIMPLEX
        font_text = cv2.FONT_HERSHEY_DUPLEX

        title = "SPEED MONITORING"
        title_size = cv2.getTextSize(title, font_title, 0.7, 2)[0]
        title_x = panel_x + (self.panel_width - title_size[0]) // 2
        title_y = panel_y + 35
        cv2.putText(img, title, (title_x, title_y), font_title, 0.7, (0, 100, 255), 2)

        sep_y = title_y + 10
        cv2.line(img, (panel_x + 20, sep_y), (panel_x + self.panel_width - 20, sep_y), (0, 0, 200), 1)

        limit_text = f"Speed limit: {self.speed_limit} km/h"
        limit_y = sep_y + 30
        cv2.putText(img, limit_text, (panel_x + 20, limit_y), font_text, 0.8, (200, 200, 200), 1)

        count_text = f"Speeding: {self.speeding_count}"
        count_size = cv2.getTextSize(count_text, font_text, 0.7, 2)[0]
        count_x = panel_x + (self.panel_width - count_size[0]) // 2
        count_y = limit_y + 30
        cv2.putText(img, count_text, (count_x, count_y), font_text, 0.6, (0, 0, 255), 2)

        vehicle_text = f"Vehicles: {self.total_count}"
        vy = count_y + 30
        cv2.putText(img, vehicle_text, (panel_x + 20, vy), font_text, 0.7, (200, 200, 200), 1)

        types_text = f"C:{self.count_by_type['car']}  T:{self.count_by_type['truck']}  B:{self.count_by_type['bus']} By:{self.count_by_type['bicycle']}  M:{self.count_by_type['motorcycle']} T:{self.count_by_type['train']} "
        cv2.putText(img, types_text, (panel_x + 20, vy + 30), font_text, 0.7, (200, 200, 200), 1)

        jam_y = vy + 70
        if self.jam_detected:
            cv2.putText(img, "⚠️ TRAFFIC JAM DETECTED", (panel_x + 20, jam_y), font_text, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Traffic: Normal", (panel_x + 20, jam_y), font_text, 0.8, (0, 255, 0), 1)

    # ------------------- Boîte véhicule -------------------
    def draw_vehicle_box(self, img, box, color, label, speed=None):
        x1, y1, x2, y2 = map(int, box)
        label_color = (0, 0, 255) if speed and speed > self.speed_limit else (0, 255, 255)
        label_text = f"{int(speed)} km/h (!)" if speed and speed > self.speed_limit else f"{int(speed)} km/h" if speed else label

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        corner_len = 12
        thickness = 2
        corners = [
            ((x1, y1), (x1 + corner_len, y1)),
            ((x1, y1), (x1, y1 + corner_len)),
            ((x2, y1), (x2 - corner_len, y1)),
            ((x2, y1), (x2, y1 + corner_len)),
            ((x1, y2), (x1 + corner_len, y2)),
            ((x1, y2), (x1, y2 - corner_len)),
            ((x2, y2), (x2 - corner_len, y2)),
            ((x2, y2), (x2, y2 - corner_len))
        ]
        for pt1, pt2 in corners:
            cv2.line(img, pt1, pt2, color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(label_text, font, font_scale, 1)[0]
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > text_size[1] else y2 + text_size[1] + 10
        cv2.rectangle(img, (label_x, label_y - text_size[1] - 4),
                      (label_x + text_size[0] + 6, label_y + 4), label_color, -1)
        cv2.putText(img, label_text, (label_x + 3, label_y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # ------------------- Estimation vitesse -------------------
    def estimate_speed(self, im0, tracks):
        if tracks[0].boxes.id is None:
            self.draw_speed_panel(im0)
            return im0

        self.draw_dashed_line(im0, self.reg_pts[0], self.reg_pts[1], color=(0, 0, 0), thickness=self.tf)

        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        t_ids = tracks[0].boxes.id.int().cpu().tolist()

        current_speeding_ids = set()

        for box, t_id, cls in zip(boxes, t_ids, clss):
            track = self.trk_history[t_id]
            bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
            track.append(bbox_center)
            if len(track) > 30: track.pop(0)

            trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            name = self.names[int(cls)].lower()
            
            color = (255, 0, 0) if name == "car" \
                else (255, 128, 0) if name == "bus" \
                else (0, 140, 255) if name == "truck" \
                else (0, 255, 0) if name == "train" \
                else (255, 255, 0) if name == "motorcycle" \
                else (255, 0, 255) if name == "bicycle" \
                else (100, 100, 100) # Couleur par défaut pour les autres classes

            if t_id not in self.seen_ids:
                self.seen_ids.add(t_id)
                self.total_count += 1
                if name in self.count_by_type: 
                    self.count_by_type[name] += 1
                
                
            # Calcul de vitesse en km/h
            if t_id in self.trk_pp:
                dx = (track[-1][0] - self.trk_pp[t_id][0]) * self.pixels_to_meters
                dy = (track[-1][1] - self.trk_pp[t_id][1]) * self.pixels_to_meters
                dist_m = np.sqrt(dx**2 + dy**2)
                dt = time() - self.trk_pt.get(t_id, time())
                if dt > 0:
                    speed_kmh = (dist_m / dt) * 3.6
                    self.spd[t_id] = speed_kmh
            self.trk_pt[t_id] = time()
            self.trk_pp[t_id] = track[-1]

            current_speed = int(self.spd.get(t_id, 0))
            self.draw_vehicle_box(im0, box, color, self.names[int(cls)].capitalize(), current_speed)

            if current_speed > self.speed_limit:
                current_speeding_ids.add(t_id)

        new_speeding = current_speeding_ids - self.counted_ids
        self.speeding_count += len(new_speeding)
        self.counted_ids.update(current_speeding_ids)

        # Historique vitesses
        if self.spd:
            mean_speed = sum(self.spd.values()) / len(self.spd)
            self.speed_history.append(mean_speed)
            if len(self.speed_history) > self.jam_window:
                self.speed_history.pop(0)

        # Détection embouteillage
        self.detect_traffic_jam(boxes)

        # Affichage panneau
        self.draw_speed_panel(im0)

        if self.view_img and self.env_check:
            cv2.imshow("Speed Monitoring", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"): return

        return im0


# ------------------- Main -------------------
if __name__ == "__main__":
    vehicle_classes = {
        1: "bicycle", 
        2: "car", 
        3: "motorcycle", 
        5: "bus", 
        7: "truck",
        6: "train"
    }
    estimator = SpeedEstimator(vehicle_classes, view_img=True)
