import cv2
import numpy as np
import time


def detect_ad_fast_high_res(video_path, ad_image_path):
    template = cv2.imread(ad_image_path, 0)
    cap = cv2.VideoCapture(video_path)
    
    W_ORIG = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_ORIG = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 24
    TOTAL_AREA = W_ORIG * H_ORIG

    # Cấu hình tối ưu
    DOWN_SCALE = 0.25  # Thu nhỏ bản nháp
    MATCH_THRESHOLD = 0.75
    
    # Biến trạng thái để tracking
    last_loc = None
    last_scale = None
    top_matches = [] # Lưu danh sách Top 5 kết quả cao nhất

    frame_count = 0
    start_time = time.time()

    print(f"--- Đang xử lý: {W_ORIG}x{H_ORIG} | FPS: {FPS:.1f} ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # TỐI ƯU: Chỉ xử lý mỗi 5 frame (0.2 giây/lần)
        if frame_count % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found_this_frame = None

        # --- CHIẾN THUẬT QUÉT ---
        # 1. Nếu đã biết vị trí trước đó (Tracking), quét vùng nhỏ xung quanh (Cực nhanh)
        if last_loc:
            x, y, w, h = last_loc
            padding = 100
            roi_y1, roi_y2 = max(0, y-padding), min(H_ORIG, y+h+padding)
            roi_x1, roi_x2 = max(0, x-padding), min(W_ORIG, x+w+padding)
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Quét tinh trong vùng ROI
            t_fine = cv2.resize(template, (w, h))
            if roi.shape[0] >= h and roi.shape[1] >= w:
                res = cv2.matchTemplate(roi, t_fine, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > MATCH_THRESHOLD:
                    found_this_frame = (max_val, (roi_x1 + max_loc[0], roi_y1 + max_loc[1]), w, h)

        # 2. Nếu không tìm thấy bằng Tracking, quét toàn bộ (Global Search)
        if not found_this_frame:
            small_gray = cv2.resize(gray, (0,0), fx=DOWN_SCALE, fy=DOWN_SCALE)
            best_global = -1
            # Quét scale thưa để tăng tốc
            for scale in np.linspace(0.3, 1.5, 30):
                sw = int(template.shape[1] * scale * DOWN_SCALE)
                sh = int(template.shape[0] * scale * DOWN_SCALE)
                if sh > small_gray.shape[0] or sw > small_gray.shape[1]: continue
                
                t_small = cv2.resize(template, (sw, sh))
                res = cv2.matchTemplate(small_gray, t_small, cv2.TM_CCOEFF_NORMED)
                _, m_val, _, m_loc = cv2.minMaxLoc(res)
                
                if m_val > best_global:
                    best_global = m_val
                    # Quy đổi ngược về tọa độ gốc
                    real_w, real_h = int(template.shape[1] * scale), int(template.shape[0] * scale)
                    found_this_frame = (m_val, (int(m_loc[0]/DOWN_SCALE), int(m_loc[1]/DOWN_SCALE)), real_w, real_h)

        # --- XỬ LÝ KẾT QUẢ ---
        if found_this_frame and found_this_frame[0] > MATCH_THRESHOLD:
            score, (rx, ry), rw, rh = found_this_frame
            area_p = (rw * rh / TOTAL_AREA) * 100
            timestamp = frame_count / FPS
            
            # Cập nhật tracking cho frame tới
            last_loc = (rx, ry, rw, rh)

            # Chỉ lưu vào Top 5 nếu score thực sự cao và không bị trùng lặp quá nhiều
            if len(top_matches) < 5 or score > top_matches[-1][0]:
                top_matches.append((score, timestamp, area_p))
                top_matches = sorted(top_matches, key=lambda x: x[0], reverse=True)[:5]
                
                # Chỉ in ra khi tìm thấy mốc "đỉnh" mới để tránh spam console
                print(f"✨ [Giây {timestamp:.2f}] New Top Match: {score*100:.1f}% | Area: {area_p:.2f}%")

            # Vẽ nhanh lên frame hiển thị (đã resize)
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0,255,0), 10)
        else:
            last_loc = None # Mất dấu, frame sau sẽ quét toàn bộ lại

        # Hiển thị preview 720p (có thể tắt dòng này để chạy nhanh hơn nữa)
        disp = cv2.resize(frame, (1280, 720))
        cv2.imshow("Monitoring", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- BÁO CÁO TOP 5 ĐỘ KHỚP CAO NHẤT ---")
    for i, (s, t, a) in enumerate(top_matches):
        print(f"{i+1}. Giây: {t:.2f}s | Độ khớp: {s*100:.1f}% | Diện tích: {a:.2f}%")


def detect_ad_comprehensive(video_path, ad_image_path):
    # 1. Tải ảnh mẫu (grayscale)
    template = cv2.imread(ad_image_path, 0)
    if template is None:
        print("❌ Lỗi: Không tìm thấy file ảnh mẫu! Kiểm tra đường dẫn.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 24 
    
    # Thông số chuẩn 720p
    V_W, V_H = 1280, 720
    TOTAL_AREA = V_W * V_H

    print(f"--- Đang bắt đầu phân tích Video ({V_W}x{V_H}) ---")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        # Resize frame về chuẩn để tính % diện tích chính xác
        frame = cv2.resize(frame, (V_W, V_H))
        
        # TỐI ƯU: Chỉ quét Ad mỗi 5 frame để tăng tốc độ xử lý (FPS)
        if frame_count % 5 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            best_match = None

            # 2. Quét đa tỉ lệ (Multi-scale)
            # Thử nghiệm ảnh mẫu từ 30% đến 80% kích thước gốc
            for scale in np.linspace(0.3, 0.8, 10):
                w = int(template.shape[1] * scale)
                h = int(template.shape[0] * scale)
                
                if h > V_H or w > V_W: continue
                
                resized_tpl = cv2.resize(template, (w, h))
                res = cv2.matchTemplate(gray_frame, resized_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if best_match is None or max_val > best_match[0]:
                    best_match = (max_val, max_loc, w, h)

            # 3. Bộ lọc logic: Ngưỡng khớp > 0.75 và diện tích ~25%
            if best_match and best_match[0] > 0.75:
                score, loc, tw, th = best_match
                percentage = (tw * th / TOTAL_AREA) * 100
                
                # Chỉ xử lý nếu diện tích nằm trong khoảng mong muốn (15% - 35%)
                if 15 <= percentage <= 35:
                    timestamp = frame_count / fps
                    print(f"✅ [GIÂY {timestamp:.2f}] Ad Detected! Khớp: {score*100:.1f}% | Area: {percentage:.2f}%")
                    
                    # --- SỬA LỖI VẼ KHUNG Ở ĐÂY ---
                    top_left = loc
                    bottom_right = (top_left[0] + tw, top_left[1] + th)
                    
                    # Vẽ khung xanh
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
                    
                    # Ghi text (tránh văng khỏi màn hình nếu Ad ở sát mép trên)
                    text_y = top_left[1] - 10 if top_left[1] > 20 else top_left[1] + 25
                    cv2.putText(frame, f"Ad: {percentage:.1f}%", (top_left[0], text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hiển thị video
        cv2.imshow('Ad Detector System', frame)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- Hoàn thành phân tích ---")


def detect_ad_comprehensive_4k(video_path, ad_image_path):
    # 1. Tải ảnh mẫu (Template) - Chuyển sang Grayscale để so khớp
    template_img = cv2.imread(ad_image_path)
    if template_img is None:
        print("❌ Lỗi: Không tìm thấy file ảnh mẫu!")
        return
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    tH_orig, tW_orig = template_gray.shape[:2]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24

    # Lấy thông số Video 4K gốc
    ORIGINAL_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ORIGINAL_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL_AREA_4K = ORIGINAL_W * ORIGINAL_H

    # Tỉ lệ thu nhỏ để xử lý (Scale Down để tăng tốc quét)
    process_scale = 0.25 
    SCAN_W = int(ORIGINAL_W * process_scale)
    SCAN_H = int(ORIGINAL_H * process_scale)

    print(f"--- Đang phân tích Video 4K: {ORIGINAL_W}x{ORIGINAL_H} ---")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # Chỉ quét mỗi 5 frame để giữ FPS cao
        if frame_count % 5 == 0:
            small_frame = cv2.resize(frame, (SCAN_W, SCAN_H))
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            best_match = None # Reset cho mỗi lần quét

            # 2. Quét đa tỉ lệ (Multi-scale)
            for scale in np.linspace(0.2, 1.5, 30): 
                # Resize template tỉ lệ thuận với bản preview nhỏ
                w = int(tW_orig * scale * process_scale)
                h = int(tH_orig * scale * process_scale)
                
                if h > SCAN_H or w > SCAN_W or h < 20: continue
                
                resized_tpl = cv2.resize(template_gray, (w, h))
                
                # THỰC HIỆN TÍNH TOÁN ĐỘ KHỚP
                res = cv2.matchTemplate(gray_small, resized_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                # Lưu lại kết quả có độ khớp (max_val) cao nhất
                if best_match is None or max_val > best_match[0]:
                    best_match = (max_val, max_loc, scale, w, h)

            # 3. Hiển thị nếu vượt ngưỡng (ví dụ > 0.7)
            if best_match and best_match[0] > 0.70:
                score, loc_small, scale_found, sw, sh = best_match
                
                # Quy đổi về kích thước và tọa độ 4K thật
                real_tw = int(tW_orig * scale_found)
                real_th = int(tH_orig * scale_found)
                top_left = (int(loc_small[0] / process_scale), int(loc_small[1] / process_scale))
                bottom_right = (top_left[0] + real_tw, top_left[1] + real_th)
                
                # Tỉ lệ diện tích thực trên khung hình 4K
                percentage = (real_tw * real_th / TOTAL_AREA_4K) * 100
                
                # Vẽ khung và thông số lên frame gốc (4K)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 10)
                
                # Hiển thị Score (Độ khớp) và Area (%)
                # Score * 100 để ra tỉ lệ % giống nhau
                status_text = f"Match: {score*100:.1f}% | Area: {percentage:.2f}%"
                cv2.putText(frame, status_text, (top_left[0], top_left[1] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
                
                timestamp = frame_count / fps
                print(f"📍 [{timestamp:.1f}s] {status_text}")

        # Hiển thị Preview 720p để quan sát
        display_res = cv2.resize(frame, (1280, 720))
        cv2.imshow('4K Ad Detection (Monitoring)', display_res)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def test_overlay_on_image(background_path, overlay_path):
    # 1. Đọc ảnh
    bg_img = cv2.imread(background_path)
    overlay_tpl = cv2.imread(overlay_path) # Đọc màu để xử lý chính xác hơn nếu cần

    if bg_img is None or overlay_tpl is None:
        print("❌ Lỗi: Không tìm thấy file ảnh.")
        return

    # Resize ảnh nền về chuẩn 720p để đồng nhất tỉ lệ tính toán
    bg_img = cv2.resize(bg_img, (1280, 720))
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    
    # Template cũng cần chuyển sang Gray
    template = cv2.cvtColor(overlay_tpl, cv2.COLOR_BGR2GRAY)
    (tH, tW) = template.shape[:2]
    
    V_W, V_H = 1280, 720
    TOTAL_AREA = V_W * V_H
    found = None

    print("--- Đang phân tích đa tỉ lệ ---")

    # 2. Quét đa tỉ lệ: Thay vì resize template, ta resize ảnh nền (hiệu quả hơn)
    # Hoặc resize template từ 20% đến 150% kích thước gốc để cover mọi trường hợp
    for scale in np.linspace(0.2, 1.5, 50): 
        resized_w = int(tW * scale)
        resized_h = int(tH * scale)

        # Nếu template sau khi scale to hơn ảnh nền thì bỏ qua
        if resized_h > V_H or resized_w > V_W:
            continue

        resized_tpl = cv2.resize(template, (resized_w, resized_h))
        
        # Sử dụng TM_CCOEFF_NORMED là thuật toán mạnh nhất cho ảnh tĩnh
        res = cv2.matchTemplate(bg_gray, resized_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if found is None or max_val > found[0]:
            found = (max_val, max_loc, scale, resized_w, resized_h)

    # 3. Hiển thị kết quả
    THRESHOLD = 0.65  # Giảm xuống 0.65 nếu ảnh thực tế có nhiễu hoặc ánh sáng khác
    if found and found[0] >= THRESHOLD:
        score, loc, scale, tw, th = found
        percentage = (tw * th / TOTAL_AREA) * 100
        
        top_left = loc
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        
        print(f"✅ Độ khớp: {score*100:.1f}%")
        print(f"✅ Tỉ lệ diện tích: {percentage:.2f}% (Scale: {scale:.2f})")

        # Vẽ kết quả
        result_display = bg_img.copy()
        cv2.rectangle(result_display, top_left, bottom_right, (0, 255, 0), 3)
        
        label = f"Match: {score*100:.1f}% | Area: {percentage:.1f}%"
        cv2.putText(result_display, label, (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Result", result_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        max_score = found[0] * 100 if found else 0
        print(f"❌ Không tìm thấy. Độ khớp tốt nhất: {max_score:.1f}%")


def test_overlay_4k(background_path, overlay_path):
    # 1. Đọc ảnh gốc (Giữ nguyên độ phân giải 4K)
    bg_img = cv2.imread(background_path)
    overlay_tpl = cv2.imread(overlay_path)

    if bg_img is None or overlay_tpl is None:
        return

    # Lấy thông số gốc 4K
    H_4K, W_4K = bg_img.shape[:2]
    TOTAL_AREA_4K = W_4K * H_4K

    # 2. Tạo bản Preview thấp (ví dụ: chia 4) để tìm kiếm nhanh
    # Điều này giúp giảm khối lượng tính toán đi 16 lần (4x4)
    scale_down = 0.25 
    bg_small = cv2.resize(bg_img, (0, 0), fx=scale_down, fy=scale_down)
    bg_gray_small = cv2.cvtColor(bg_small, cv2.COLOR_BGR2GRAY)
    
    template_gray = cv2.cvtColor(overlay_tpl, cv2.COLOR_BGR2GRAY)
    tH, tW = template_gray.shape[:2]

    best_match = None

    print(f"--- Đang quét trên bản thu nhỏ (Fast Scan) ---")
    
    # Quét đa tỉ lệ trên bản nhỏ
    for scale in np.linspace(0.1, 2.0, 40):
        # Resize template tương ứng với bản thu nhỏ
        curr_w = int(tW * scale * scale_down)
        curr_h = int(tH * scale * scale_down)

        if curr_h > bg_gray_small.shape[0] or curr_w > bg_gray_small.shape[1] or curr_h < 10:
            continue

        resized_tpl = cv2.resize(template_gray, (curr_w, curr_h))
        res = cv2.matchTemplate(bg_gray_small, resized_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if best_match is None or max_val > best_match[0]:
            best_match = (max_val, max_loc, scale)

    # 3. Ánh xạ kết quả ngược lại ảnh 4K
    if best_match and best_match[0] > 0.65:
        score, loc_small, scale_found = best_match
        
        # Tính toán tọa độ và kích thước trên ảnh 4K gốc
        # Tọa độ gốc = Tọa độ nhỏ / scale_down
        real_w = int(tW * scale_found)
        real_h = int(tH * scale_found)
        top_left = (int(loc_small[0] / scale_down), int(loc_small[1] / scale_down))
        bottom_right = (top_left[0] + real_w, top_left[1] + real_h)

        # Tính % diện tích chuẩn trên 4K
        percentage = (real_w * real_h / TOTAL_AREA_4K) * 100

        print(f"✅ Tìm thấy trên ảnh 4K!")
        print(f"✅ Tỉ lệ diện tích thực: {percentage:.4f}%")
        print(f"✅ Độ khớp: {score*100:.1f}%")

        # Hiển thị (Resize kết quả để xem được trên màn hình thường)
        cv2.rectangle(bg_img, top_left, bottom_right, (0, 255, 0), 8)
        result_view = cv2.resize(bg_img, (1280, 720)) # Resize chỉ để xem
        cv2.imshow("4K Detection Result (Scaled View)", result_view)
        cv2.waitKey(0)
    else:
        print("❌ Không tìm thấy overlay trên ảnh 4K.")

cv2.destroyAllWindows()


if __name__ == "__main__":
    # detect_ad_comprehensive('video.mp4', 'overlay_ad_1.png')
    # detect_ad_fast_high_res('video.mp4', 'overlay_ad_1.png')
    # detect_ad_comprehensive_4k('video.mp4', 'overlay_ad_1.png')

    test_overlay_on_image('background_image.png', 'overlay_ad_2.png')
