from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import numpy as np
import uuid
import base64
import requests
from io import BytesIO
from werkzeug.utils import secure_filename
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Setup Flask
app = Flask(__name__)
app.secret_key = "very_secret_key_123"

# Cloudinary config
cloudinary.config(
    cloud_name="duk8odqun",
    api_key="485926927892748",
    api_secret="CSaf5iaR5cC5cl7PiyNlN_uAaSQ",
    secure=True
)


# Template filter for base64 (not needed here but for expansion)
@app.template_filter('b64encode')
def b64encode_filter(data):
    return base64.b64encode(data).decode('utf-8')

# Load image from Cloudinary URL
def load_image_from_url(url):
    response = requests.get(url)
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    return image

# === Your custom methods ===
from anh_2.main import stitch_images as stitch_two
from anh_2.utils import detect_and_match
from anh_nhieu.main import stitch_multiple
from anh_nhieu.utils import load_images_from_urls

@app.route("/", methods=["GET", "POST"])
def index():
    result_url = None
    match_mode = False
    error_message = None

    # F5 hoặc lần đầu: reset session
    if request.method == "GET" and not request.referrer:
        session.clear()

    uploaded_urls = session.get("uploaded_urls", [])

    if request.method == "POST":
        action = request.form.get("mode")
        filenames = uploaded_urls
        stitched_image = None

        try:
            if action == "match":
                if len(filenames) != 2:
                    raise ValueError("❌ Exactly 2 images are required to show matching.")
                img1 = load_image_from_url(filenames[0])
                img2 = load_image_from_url(filenames[1])
                kp1, kp2, matches = detect_and_match(img1, img2)
                if len(matches) < 4:
                    raise ValueError("❌ Not enough matching points between the images.")
                stitched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                match_mode = True

            elif action == "stitch2":
                if len(filenames) != 2:
                    raise ValueError("❌ Exactly 2 images are required for stitching.")
                img1 = load_image_from_url(filenames[0])
                img2 = load_image_from_url(filenames[1])
                kp1, kp2, matches = detect_and_match(img1, img2)
                if len(matches) < 4:
                    raise ValueError("❌ Not enough matching points to perform stitching.")
                stitched_image = stitch_two(img1, img2, kp1, kp2, matches)

            elif action == "stitchn":
                if len(filenames) <= 2:
                    raise ValueError("❌ Only 2 images – please choose 'Stitch 2 Images' instead.")
                imgs = [load_image_from_url(url) for url in filenames]
                stitched_image = stitch_multiple(imgs)

            else:
                raise ValueError("❌ Invalid action.")

            if stitched_image is None or np.count_nonzero(stitched_image) < 10:
                raise ValueError("❌ Stitching failed – images may not be similar.")

            # Encode ảnh và upload lên Cloudinary
            _, buffer = cv2.imencode(".png", stitched_image)
            upload_result = cloudinary.uploader.upload(
                BytesIO(buffer.tobytes()),
                folder="images-stitching",
                public_id=str(uuid.uuid4())
            )
            result_url = upload_result["secure_url"]
            session['result_url'] = result_url

        except Exception as e:
            error_message = str(e)

        return render_template("index.html",
                               result_url=session.get("result_url"),
                               uploaded_images=uploaded_urls,
                               match_mode=match_mode,
                               error_message=error_message)

    return render_template("index.html",
                           result_url=session.get("result_url"),
                           uploaded_images=uploaded_urls,
                           match_mode=False,
                           error_message=None)

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images")
    uploaded_urls = session.get("uploaded_urls", [])
    for file in files:
        upload_result = cloudinary.uploader.upload(
            file,
            folder="images-stitching",
            public_id=str(uuid.uuid4())
        )
        uploaded_urls.append(upload_result["secure_url"])
    session['uploaded_urls'] = uploaded_urls
    return redirect(url_for("index"))

@app.route("/delete/<int:index>")
def delete_image(index):
    uploaded_urls = session.get("uploaded_urls", [])
    if 0 <= index < len(uploaded_urls):
        del uploaded_urls[index]
    session['uploaded_urls'] = uploaded_urls
    return redirect(url_for("index"))

import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)

