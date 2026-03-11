import cv2
import numpy as np
import json


class OverlayAdDetector:

    def __init__(self):

        self.sample_interval = 5
        self.required_hits = 3
        self.min_matches = 8

        self.orb = cv2.ORB_create(
            nfeatures=3000,
            scaleFactor=1.2,
            nlevels=8
        )

    # ------------------------------------------------

    def extract_features(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, None)

        return kp, des

    # ------------------------------------------------

    def match_features(self, des1, des2):

        if des1 is None or des2 is None:
            return []

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        matches = bf.knnMatch(des1, des2, k=2)

        good = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        return good

    # ------------------------------------------------

    def build_ad_templates(self, ad_frame):

        scales = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15]

        templates = []

        for s in scales:

            w = int(ad_frame.shape[1] * s)
            h = int(ad_frame.shape[0] * s)

            if w < 60 or h < 40:
                continue

            resized = cv2.resize(ad_frame, (w, h))

            kp, des = self.extract_features(resized)

            templates.append({
                "image": resized,
                "kp": kp,
                "des": des,
                "shape": resized.shape[:2]
            })

        return templates

    # ------------------------------------------------

    def detect_ad(self, frame, templates):

        kp, des = self.extract_features(frame)

        best_score = 0
        best_box = None

        for t in templates:

            matches = self.match_features(t["des"], des)

            if len(matches) < self.min_matches:
                continue

            src_pts = np.float32(
                [t["kp"][m.queryIdx].pt for m in matches]
            ).reshape(-1,1,2)

            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]
            ).reshape(-1,1,2)

            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                5.0
            )

            if H is None:
                continue

            inliers = mask.ravel().sum()

            if inliers < 6:
                continue

            h, w = t["shape"]

            corners = np.float32([
                [0,0],
                [w,0],
                [w,h],
                [0,h]
            ]).reshape(-1,1,2)

            projected = cv2.perspectiveTransform(corners, H)

            xs = projected[:,0,0]
            ys = projected[:,0,1]

            x = int(xs.min())
            y = int(ys.min())
            bw = int(xs.max() - xs.min())
            bh = int(ys.max() - ys.min())

            if bw < 40 or bh < 30:
                continue

            if inliers > best_score:

                best_score = inliers
                best_box = (x,y,bw,bh)

        if best_box is None:
            return False, None, 0

        return True, best_box, best_score

    # ------------------------------------------------

    def compute_coverage(self, box, frame_shape):

        h, w = frame_shape

        x,y,bw,bh = box

        frame_area = w * h
        ad_area = bw * bh

        return (ad_area / frame_area) * 100

    # ------------------------------------------------

    def compute_position(self, box, frame_shape):

        h, w = frame_shape

        x,y,bw,bh = box

        cx = x + bw/2
        cy = y + bh/2

        if cx < w/2 and cy < h/2:
            return "top-left"

        if cx >= w/2 and cy < h/2:
            return "top-right"

        if cx < w/2 and cy >= h/2:
            return "bottom-left"

        return "bottom-right"

    # ------------------------------------------------

    def analyze(self, main_video, ad_video):

        cap = cv2.VideoCapture(main_video)

        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_shape = (frame_height, frame_width)

        ad_cap = cv2.VideoCapture(ad_video)
        ret, ad_frame = ad_cap.read()
        ad_cap.release()

        templates = self.build_ad_templates(ad_frame)

        frame_id = 0

        hits = []
        scores = []
        boxes = []

        consecutive = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % self.sample_interval != 0:
                frame_id += 1
                continue

            found, box, score = self.detect_ad(
                frame,
                templates
            )

            if found:

                consecutive += 1

                if consecutive >= self.required_hits:

                    t = frame_id / fps

                    hits.append(t)
                    scores.append(score)
                    boxes.append(box)

            else:

                consecutive = 0

            frame_id += 1

        cap.release()

        if not hits:
            return {"ad_found": False}

        start = min(hits)
        end = max(hits)

        box = boxes[len(boxes)//2]

        coverage = self.compute_coverage(box, frame_shape)

        position = self.compute_position(box, frame_shape)

        x,y,w,h = box

        confidence = min(np.mean(scores) / 30, 1.0)

        result = {

            "ad_found": True,
            "start_time": round(start,2),
            "end_time": round(end,2),
            "duration": round(end-start,2),
            "overlay_size_px": [w,h],
            "coverage_percent": round(coverage,2),
            "position": position,
            "confidence": round(confidence,2)

        }

        return result


# ------------------------------------------------

if __name__ == "__main__":

    detector = OverlayAdDetector()

    result = detector.analyze(
        "main_video.mp4",
        "ad_video.mp4"
    )

    print(json.dumps(result, indent=2))
