from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from anh_2.main import stitch_images as stitch_two
from anh_2.utils import load_image, detect_and_match
from anh_nhieu.main import stitch_multiple
from anh_nhieu.utils import load_images

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_PATH = 'static/result.jpg'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_uploaded_images():
    return [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    match_mode = False
    error_message = None
    uploaded_images = get_uploaded_images()

    if request.method == "POST":
        action = request.form.get("mode")
        filenames = [os.path.join(UPLOAD_FOLDER, f) for f in uploaded_images]
        stitched_image = None

        try:
            if action == "match":
                if len(filenames) != 2:
                    raise ValueError("❌ Cần đúng 2 ảnh để hiển thị matching.")
                img1 = load_image(filenames[0])
                img2 = load_image(filenames[1])
                kp1, kp2, matches = detect_and_match(img1, img2)
                if len(matches) < 4:
                    raise ValueError("❌ Các ảnh không có đủ điểm tương đồng để hiển thị matching.")
                stitched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                match_mode = True

            elif action == "stitch2":
                if len(filenames) != 2:
                    raise ValueError("❌ Cần đúng 2 ảnh để ghép.")
                img1 = load_image(filenames[0])
                img2 = load_image(filenames[1])
                kp1, kp2, matches = detect_and_match(img1, img2)
                if len(matches) < 4:
                    raise ValueError("❌ Các ảnh không có đủ điểm tương đồng để ghép.")
                stitched_image = stitch_two(img1, img2, kp1, kp2, matches)
                if stitched_image is None or np.count_nonzero(stitched_image) < 10:
                    raise ValueError("❌ Không thể ghép ảnh – ảnh không phù hợp hoặc không giống nhau.")

            elif action == "stitchn":
                if len(filenames) <= 2:
                    raise ValueError("❌ Chỉ có 2 ảnh – bạn chỉ được chọn ghép 2 ảnh.")
                imgs = load_images(filenames)
                if len(imgs) < 3:
                    raise ValueError("❌ Không đủ ảnh hợp lệ để ghép.")
                stitched_image = stitch_multiple(imgs)
                if stitched_image is None or np.count_nonzero(stitched_image) < 10:
                    raise ValueError("❌ Không thể ghép ảnh – các ảnh không có điểm tương đồng.")

            else:
                raise ValueError("❌ Hành động không hợp lệ.")

            if stitched_image is not None:
                cv2.imwrite(RESULT_PATH, stitched_image)
                result_image = RESULT_PATH
            else:
                raise ValueError("❌ Không thể tạo ảnh kết quả.")

        except Exception as e:
            error_message = str(e)

        return render_template("index.html", result_image=result_image,
                               match_mode=match_mode,
                               uploaded_images=uploaded_images,
                               error_message=error_message)

    return render_template("index.html", result_image=None, uploaded_images=uploaded_images, error_message=None)

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images")
    for file in files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
    return redirect(url_for("index"))

@app.route("/delete/<filename>")
def delete_image(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return redirect(url_for("index"))

@app.route("/download")
def download():
    return send_file(RESULT_PATH, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
