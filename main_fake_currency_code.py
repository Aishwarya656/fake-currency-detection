import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os


# ---------- Improved SIFT matching with homography ----------
def match_features(img1, img2, ratio=0.7):
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    # FLANN matcher setup
    flann = cv.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [
        m for m, n in matches
        if m.distance < ratio * n.distance
    ]

    # Check geometric consistency using homography
    if len(good) > 10:
        src_pts = np.float32([
            kp1[m.queryIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp2[m.trainIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(
            src_pts,
            dst_pts,
            cv.RANSAC,
            5.0
        )

        if mask is None:
            return 0

        inliers = np.sum(mask)
        return inliers

    return 0


# ---------- Currency detection function ----------
def detect_currency(denomination, ref_front, ref_back=None):

    print(
        f"Checking reference paths:\n"
        f"Front: {ref_front}\n"
        f"Back: {ref_back}"
    )

    if not os.path.exists(ref_front):
        messagebox.showerror(
            "Error",
            f"Front reference image for {denomination} not found!\n"
            "Check dataset path."
        )
        return

    if ref_back and not os.path.exists(ref_back):
        print(
            f"Back reference image for {denomination} "
            "not found — disabling back check."
        )
        ref_back = None

    def check(side):

        if side == "Back" and not ref_back:
            messagebox.showwarning(
                "Unavailable",
                f"No back reference image available for {denomination}."
            )
            return

        path = filedialog.askopenfilename(
            title=f"Select {denomination} {side} Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp")
            ]
        )

        if not path or not os.path.exists(path):
            messagebox.showerror(
                "Error",
                "Selected image not found!"
            )
            return

        img_test = cv.imread(path)

        img_ref = cv.imread(
            ref_front if side == "Front" else ref_back
        )

        if img_test is None or img_ref is None:
            messagebox.showerror(
                "Error",
                "Failed to load images!"
            )
            return

        # Resize test image to match reference
        img_test = cv.resize(
            img_test,
            (img_ref.shape[1], img_ref.shape[0])
        )

        # Convert to grayscale and equalize histogram
        img_test_gray = cv.cvtColor(
            img_test,
            cv.COLOR_BGR2GRAY
        )

        img_ref_gray = cv.cvtColor(
            img_ref,
            cv.COLOR_BGR2GRAY
        )

        img_test_gray = cv.equalizeHist(
            img_test_gray
        )

        img_ref_gray = cv.equalizeHist(
            img_ref_gray
        )

        # Perform feature matching
        count = match_features(
            img_ref_gray,
            img_test_gray
        )

        print(
            f"{denomination} {side} "
            f"Inlier Matches: {count}"
        )

        # Thresholds for decision
        if denomination == "₹2000":
            threshold = 180

        elif denomination == "₹500":
            threshold = 120

        else:
            threshold = 100

        # Display result
        if count > threshold:
            messagebox.showinfo(
                "Result",
                f"{denomination} {side} is REAL!\n"
                f"Inlier Matches: {count}"
            )

        else:
            messagebox.showwarning(
                "Result",
                f"{denomination} {side} is FAKE!\n"
                f"Inlier Matches: {count}"
            )

    # ---------- Sub-window for front/back check ----------
    top = tk.Toplevel()
    top.title(f"{denomination} Detection")
    top.geometry("300x220")

    tk.Label(
        top,
        text=f"Check {denomination} Currency",
        font=("Arial", 14)
    ).pack(pady=10)

    tk.Button(
        top,
        text="Check Front Side",
        fg="green",
        width=20,
        command=lambda: check("Front")
    ).pack(pady=5)

    if ref_back:
        tk.Button(
            top,
            text="Check Back Side",
            fg="blue",
            width=20,
            command=lambda: check("Back")
        ).pack(pady=5)

    tk.Button(
        top,
        text="Close",
        width=20,
        fg="red",
        command=top.destroy
    ).pack(pady=10)


# ---------- Main GUI ----------
root = tk.Tk()

root.title("Fake Currency Detection")
root.geometry("420x380")
root.configure(bg="white")

tk.Label(
    root,
    text="FAKE CURRENCY DETECTION",
    font=("Arial Black", 18),
    bg="white"
).pack(pady=15)

tk.Label(
    root,
    text="Select Currency to Check",
    font=("Arial", 14),
    bg="white"
).pack(pady=5)


# ---------- Dataset Path ----------
# The dataset folder should be inside the project directory.
dataset_path = os.path.join(
    os.path.dirname(__file__),
    "dataset",
    "test"
)


# ---------- ₹500 ----------
tk.Button(
    root,
    text="₹500",
    width=25,
    command=lambda: detect_currency(
        "₹500",
        os.path.join(dataset_path, "500front.jpg"),
        os.path.join(dataset_path, "500back.jpg")
    )
).pack(pady=5)


# ---------- ₹2000 ----------
tk.Button(
    root,
    text="₹2000",
    width=25,
    command=lambda: detect_currency(
        "₹2000",
        os.path.join(dataset_path, "2000front.jpg"),
        os.path.join(dataset_path, "2000back.jpg")
    )
).pack(pady=5)


# ---------- Exit button ----------
tk.Button(
    root,
    text="Exit",
    width=25,
    command=root.destroy,
    fg="red"
).pack(pady=20)


# ---------- Run application ----------
root.mainloop()