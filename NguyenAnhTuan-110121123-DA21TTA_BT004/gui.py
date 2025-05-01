import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from anh_2.main import stitch_images as stitch_two_images
from anh_2.utils import load_image as load_img_2, detect_and_match
from anh_nhieu.main import stitch_multiple
from anh_nhieu.utils import load_images as load_imgs_multi

class ImageStitchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧵 Image Stitching Tool")
        self.image_paths = []
        self.thumbnail_refs = []
        self.last_result = None

        canvas = tk.Canvas(root, bg="#f8f9fa", highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        outer_frame = tk.Frame(canvas, bg="#f8f9fa")
        canvas.create_window((0, 0), window=outer_frame, anchor="nw")

        self.inner_frame = tk.Frame(outer_frame, bg="#f8f9fa")
        self.inner_frame.pack(anchor="center", pady=20)

        tk.Label(self.inner_frame, text="📂 Chọn ảnh để ghép", font=("Segoe UI", 18, "bold"),
                 bg="#f8f9fa", fg="#212529").pack(pady=10)

        tk.Button(self.inner_frame, text="📁 Chọn ảnh", command=self.select_images,
                  bg="#0d6efd", fg="white", font=("Segoe UI", 12, "bold"),
                  activebackground="#0b5ed7", relief="flat", width=20).pack(pady=5)

        self.mode_var = tk.StringVar(value="2")
        radio_frame = tk.Frame(self.inner_frame, bg="#f8f9fa")
        radio_frame.pack(pady=5)
        tk.Radiobutton(radio_frame, text="🖼️ Ghép 2 ảnh", variable=self.mode_var, value="2",
                       bg="#f8f9fa", fg="#495057", font=("Segoe UI", 11), selectcolor="#dbe4ff").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(radio_frame, text="🖼️ Ghép nhiều ảnh", variable=self.mode_var, value="n",
                       bg="#f8f9fa", fg="#495057", font=("Segoe UI", 11), selectcolor="#dbe4ff").pack(side=tk.LEFT, padx=10)

        tk.Button(self.inner_frame, text="🔍 Hiển thị matching", command=self.show_matching,
                  bg="#d63384", fg="white", font=("Segoe UI", 12, "bold"),
                  activebackground="#c2255c", relief="flat", width=20).pack(pady=5)

        tk.Button(self.inner_frame, text="🧵 Thực hiện ghép ảnh", command=self.stitch_images,
                  bg="#198754", fg="white", font=("Segoe UI", 12, "bold"),
                  activebackground="#157347", relief="flat", width=20).pack(pady=5)

        tk.Button(self.inner_frame, text="💾 Lưu ảnh kết quả", command=self.save_result,
                  bg="#ffc107", fg="black", font=("Segoe UI", 12, "bold"),
                  activebackground="#ffca2c", relief="flat", width=20).pack(pady=5)

        self.thumbnail_frame = tk.Frame(self.inner_frame, bg="#f8f9fa")
        self.thumbnail_frame.pack(pady=10)

        self.result_canvas = tk.Canvas(self.inner_frame, bg="#ffffff", width=960, height=600,
                                       bd=1, relief=tk.SUNKEN)
        self.result_canvas.pack(pady=(10, 30))

    def select_images(self):
        paths = filedialog.askopenfilenames(title="Chọn ảnh", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if paths:
            self.image_paths = list(paths)
            self.display_thumbnails()
            messagebox.showinfo("Thông báo", f"Đã chọn {len(self.image_paths)} ảnh.")

    def display_thumbnails(self):
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        self.thumbnail_refs.clear()

        for i, path in enumerate(self.image_paths):
            frame = tk.Frame(self.thumbnail_frame, bg="#f8f9fa")
            frame.pack(side=tk.LEFT, padx=5)

            img = Image.open(path)
            img.thumbnail((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            self.thumbnail_refs.append(img_tk)

            lbl = tk.Label(frame, image=img_tk, bg="#f8f9fa")
            lbl.pack()

            btn = tk.Button(frame, text="❌", fg="#dc3545", command=lambda idx=i: self.remove_image(idx),
                            bg="#f8f9fa", activebackground="#f8d7da", relief="flat", font=("Segoe UI", 10, "bold"))
            btn.pack()

    def remove_image(self, index):
        if 0 <= index < len(self.image_paths):
            del self.image_paths[index]
            self.display_thumbnails()

    def show_image(self, cv_img):
        self.last_result = cv_img.copy()
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(pil_img)

        self.result_canvas.delete("all")
        self.result_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.result_canvas.config(scrollregion=self.result_canvas.bbox(tk.ALL))
        self.result_canvas.image = img_tk

    def show_matching(self):
        if not self.image_paths or len(self.image_paths) != 2:
            messagebox.showerror("Lỗi", "Chỉ chọn đúng 2 ảnh để hiển thị matching.")
            return

        try:
            img1 = load_img_2(self.image_paths[0])
            img2 = load_img_2(self.image_paths[1])
            kp1, kp2, matches = detect_and_match(img1, img2)

            if len(matches) < 4:
                raise ValueError("Không đủ điểm matching giữa hai ảnh.")

            match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.show_image(match_img)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def stitch_images(self):
        if not self.image_paths:
            messagebox.showwarning("Cảnh báo", "Bạn chưa chọn ảnh.")
            return

        try:
            if self.mode_var.get() == "2":
                if len(self.image_paths) != 2:
                    messagebox.showerror("Lỗi", "Chỉ chọn đúng 2 ảnh.")
                    return
                img1 = load_img_2(self.image_paths[0])
                img2 = load_img_2(self.image_paths[1])
                kp1, kp2, matches = detect_and_match(img1, img2)

                if len(matches) < 4:
                    raise ValueError("Không đủ điểm matching giữa hai ảnh để ghép.")

                result = stitch_two_images(img1, img2, kp1, kp2, matches)
                self.show_image(result)
            else:
                imgs = load_imgs_multi(self.image_paths)
                result = stitch_multiple(imgs)
                self.show_image(result)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def save_result(self):
        if self.last_result is None:
            messagebox.showwarning("Chưa có ảnh", "Chưa có ảnh kết quả để lưu.")
            return

        folder = filedialog.askdirectory(title="Chọn thư mục để lưu")
        if not folder:
            return

        save_path = os.path.join(folder, "ketqua_ghep.jpg")
        cv2.imwrite(save_path, self.last_result)
        messagebox.showinfo("Đã lưu", f"Ảnh đã được lưu:\n{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageStitchApp(root)
    root.mainloop()
