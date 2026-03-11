import cv2
import numpy as np
import subprocess
import json
from PIL import Image
import imagehash
from tqdm import tqdm


#########################################
# CONFIG
#########################################

FRAME_SAMPLE_RATE = 5
SIMILARITY_THRESHOLD = 0.85
PHASH_THRESHOLD = 0.75


#########################################
# FFPROBE DURATION
#########################################

def get_duration(video):

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    data = json.loads(result.stdout)

    return float(data["format"]["duration"])


#########################################
# EXTRACT FRAMES
#########################################

def extract_frames(video_path, sample_rate=FRAME_SAMPLE_RATE):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    step = int(fps / sample_rate)

    frames = []

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % step == 0:

            frames.append(frame)

        frame_id += 1

    cap.release()

    return frames


#########################################
# HISTOGRAM SIMILARITY
#########################################

def histogram_similarity(f1, f2):

    f1 = cv2.resize(f1, (320, 240))
    f2 = cv2.resize(f2, (320, 240))

    hist1 = cv2.calcHist([f1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([f2], [0], None, [256], [0, 256])

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return score


#########################################
# PHASH SIMILARITY
#########################################

def phash_similarity(f1, f2):

    img1 = Image.fromarray(f1)
    img2 = Image.fromarray(f2)

    h1 = imagehash.phash(img1)
    h2 = imagehash.phash(img2)

    distance = h1 - h2

    similarity = 1 - distance / 64

    return similarity


#########################################
# FRAME MATCH SCORE
#########################################

def frame_score(f1, f2):

    h = histogram_similarity(f1, f2)

    p = phash_similarity(f1, f2)

    score = (h + p) / 2

    return score


#########################################
# DETECT AD START
#########################################

def detect_ad_start(playback_frames, ad_frames):

    print("Detecting ad start...")

    ad_first = ad_frames[0]

    best_index = -1
    best_score = 0

    for i in tqdm(range(len(playback_frames))):

        score = frame_score(playback_frames[i], ad_first)

        if score > best_score:

            best_score = score
            best_index = i

    print("Best match score:", best_score)

    if best_score > SIMILARITY_THRESHOLD:

        return best_index

    return None


#########################################
# COMPARE AD SEGMENT
#########################################

def compare_ads(ad_frames, playback_frames, start_index):

    scores = []

    print("Comparing frames...")

    for i in tqdm(range(len(ad_frames))):

        if start_index + i >= len(playback_frames):
            break

        f1 = ad_frames[i]
        f2 = playback_frames[start_index + i]

        s = frame_score(f1, f2)

        scores.append(s)

    avg = np.mean(scores)

    return avg


#########################################
# MAIN
#########################################

def run(original_ad, playback_video):

    print("\n----- AD TEST START -----\n")

    ad_duration = get_duration(original_ad)
    playback_duration = get_duration(playback_video)

    print("Original Ad Duration:", ad_duration)
    print("Playback Video Duration:", playback_duration)

    print("\nExtracting frames...")

    ad_frames = extract_frames(original_ad)
    playback_frames = extract_frames(playback_video)

    print("Ad frames:", len(ad_frames))
    print("Playback frames:", len(playback_frames))

    start = detect_ad_start(playback_frames, ad_frames)

    if start is None:

        print("\nAd not found in playback!")
        return

    print("\nAd detected at frame index:", start)

    score = compare_ads(ad_frames, playback_frames, start)

    print("\nAverage similarity:", score)

    print("\n----- RESULT -----")

    if score > 0.9:

        print("PASS - PERFECT MATCH")

    elif score > 0.8:

        print("PASS - ACCEPTABLE MATCH")

    else:

        print("FAIL - AD MISMATCH")


#########################################
# ENTRY
#########################################

if __name__ == "__main__":

    original_ad = "playback_ad_1.mp4"
    playback_video = "playback_video.mp4"

    run(original_ad, playback_video)
