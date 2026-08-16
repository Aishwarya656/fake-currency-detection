# Fake Currency Detection Using Image Processing

## About the Project

Fake Currency Detection is an image-processing based application developed using Python and OpenCV. The system compares a selected currency note with reference currency images and determines whether the note is likely to be real or fake.

The project uses SIFT feature extraction, FLANN-based feature matching, Lowe's ratio test, and homography-based geometric verification.

## Technologies Used

* Python
* OpenCV
* NumPy
* Tkinter
* SIFT
* FLANN

## Features

* Simple graphical user interface
* Front-side currency verification
* Back-side currency verification
* SIFT feature extraction
* FLANN-based feature matching
* Homography-based verification
* Real/fake result display
* Supports ₹500 and ₹2000 denominations

## How It Works

1. Select the currency denomination.
2. Choose the front or back side.
3. Select the image of the currency note to be tested.
4. The system compares the test image with a reference image.
5. SIFT features are extracted from both images.
6. FLANN is used to find matching features.
7. Lowe's ratio test filters the matches.
8. Homography checks geometric consistency.
9. The system displays whether the note is real or fake based on the matching result.

## Project Structure

```text
fake-currency-detection/
├── main_fake_currency_code.py
├── requirements.txt
├── README.md
└── dataset/
    └── test/
        ├── 500front.jpg
        ├── 500back.jpg
        ├── 2000front.jpg
        └── 2000back.jpg
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Run:

```bash
python main_fake_currency_code.py
```

The application will open a graphical interface where you can select the denomination and test an image.

## Future Scope

The system can be extended to support additional denominations, more currency security features, larger datasets, and improved classification techniques.

## Author

Aishwarya H
