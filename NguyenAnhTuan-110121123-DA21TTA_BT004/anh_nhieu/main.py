import cv2
import numpy as np
import os
from .utils import load_images, detect_and_match

def stitch_pair(img1, img2):
    from .utils import detect_and_match

    kp1, kp2, matches = detect_and_match(img1, img2)

    if len(matches) < 4:
        raise ValueError("Not enough similarities to calculate Homography.")
    
    # ✅ Thêm đoạn này để kiểm tra chất lượng matching
    distances = [m.distance for m in matches]
    avg_distance = np.mean(distances)
    if avg_distance > 200:
        raise ValueError("The photos are too different — they can't be matched because there's no real similarity.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    pts_img1_transformed = cv2.perspectiveTransform(pts_img1, H)
    all_pts = np.concatenate((pts_img1_transformed, pts_img2), axis=0)

    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    trans_dist = [-xmin, -ymin]

    H_trans = np.array([[1, 0, trans_dist[0]],
                        [0, 1, trans_dist[1]],
                        [0, 0, 1]])

    result = cv2.warpPerspective(img1, H_trans @ H, (xmax - xmin, ymax - ymin))
    result[trans_dist[1]:h2 + trans_dist[1], trans_dist[0]:w2 + trans_dist[0]] = img2

    return result


def stitch_multiple(images):
    stitched = images[0]
    for i in range(1, len(images)):
        stitched = stitch_pair(stitched, images[i])
    return stitched

if __name__ == "__main__":
    image_folder = "../images/"
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")])
    imgs = load_images(image_files)
    result = stitch_multiple(imgs)

    cv2.imshow("Stitched Multiple", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
