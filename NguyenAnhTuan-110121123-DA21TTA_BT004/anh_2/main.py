import cv2
import numpy as np
from .utils import load_image, detect_and_match

def draw_matches(img1, kp1, img2, kp2, matches):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def stitch_images(img1, img2, kp1, kp2, matches):
    if len(matches) < 4:
        raise ValueError("Không đủ điểm tương đồng để ghép ảnh.")
    
    # ✅ PHẢI đặt trong hàm – thụt lề đúng
    distances = [m.distance for m in matches]
    avg_distance = np.mean(distances)

    if avg_distance > 200:
        raise ValueError("Ảnh quá khác nhau — không thể ghép do không có điểm tương đồng thực sự.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None or np.isnan(H).any() or np.isinf(H).any():
        raise ValueError("Không thể tính được phép biến đổi Homography hợp lệ để ghép ảnh.")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts_img1_transformed = cv2.perspectiveTransform(pts_img1, H)
    pts_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    all_pts = np.concatenate((pts_img1_transformed, pts_img2), axis=0)

    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
    trans_dist = [-xmin, -ymin]

    H_trans = np.array([[1, 0, trans_dist[0]],
                        [0, 1, trans_dist[1]],
                        [0, 0, 1]])

    result = cv2.warpPerspective(img1, H_trans @ H, (xmax - xmin, ymax - ymin))

    if result.shape[0] < trans_dist[1] + h2 or result.shape[1] < trans_dist[0] + w2:
        raise ValueError("Kích thước ảnh không đủ để ghép ảnh thứ 2. Ghép ảnh thất bại.")

    result[trans_dist[1]:h2 + trans_dist[1], trans_dist[0]:w2 + trans_dist[0]] = img2
    return result


if __name__ == "__main__":
    img1 = load_image("../images/img1.jpg")
    img2 = load_image("../images/img2.jpg")
    kp1, kp2, good_matches = detect_and_match(img1, img2)

    if len(good_matches) < 4:
        print("Không đủ điểm matching!")
    else:
        match_img = draw_matches(img1, kp1, img2, kp2, good_matches)
        cv2.imshow("Matching", match_img)

        stitched = stitch_images(img1, img2, kp1, kp2, good_matches)
        cv2.imshow("Stitched Result", stitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
