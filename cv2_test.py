import cv2

img = cv2.imread("datasets/UIEB/input/1.PNG")  # Replace with your actual image
print("Read image:", img.shape if img is not None else "Failed")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite("test_output.jpg", blur)
print("Saved test image.")
