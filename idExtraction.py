# This code is written in PyCharm IDE

# Install libraries
import cv2
import pytesseract
import easyocr
from pytesseract import Output
import time

reader = easyocr.Reader(['tr', 'en'], gpu=True)

kimlik_arka = cv2.imread("yeni ehliyet/5_prep.jpg")

#Necessary functions for processing id images
def camPre(img):
    #Apply resize wisely what if dimensions are small??
    img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    #img = cv2.resize(img, (600, 400))
    #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img, (17, 17), 0)
    # mg = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # img = cv2.medianBlur(img, 5)
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations = 1)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # img = cv2.Canny(img, 100, 200)
    return img

# Extracting framework layout coordinates
x1 = (3123 / 5879) * kimlik_arka.shape[0]
x1 = int(x1)
x2 = (3549 / 5879) * kimlik_arka.shape[0]
x2 = int(x2)
y1 = (6519 / 9108) * kimlik_arka.shape[1]
y1 = int(y1)
y2 = (8678 / 9108) * kimlik_arka.shape[1]
y2 = int(y2)

tc = kimlik_arka[x1:x2, y1:y2, :]
tc = preprocess(camPre(tc))

# Performance comparison between Tesseract and EasyOCR
t = time.time()
custom_config = r'-l tur --oem 2 --psm 9'
words = pytesseract.image_to_string(tc, config=custom_config)
elapsed = time.time() - t
print("Tesseract completes in %s seconds: " % elapsed, words)

t = time.time()
result = reader.readtext(tc, detail=0, allowlist='0123456789')
elapsed = time.time() - t
print("EasyOCR completes in %s seconds: " % elapsed, result)

cv2.imshow("kimlik on", tc)
# Maintain output window until user presses a key
cv2.waitKey(0)

kimlik_arka = preprocess(camPre(kimlik_arka))
# Adding custom options
custom_config = r'-l tur --oem 2 --psm 4'

words = pytesseract.image_to_string(kimlik_arka, config=custom_config)
print(words)

d = pytesseract.image_to_data(kimlik_arka, config=custom_config, output_type=Output.DICT)
print(d.keys())
# Plot the boxes
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        kimlik_on_box = cv2.rectangle(kimlik_arka, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("kimlik on", kimlik_arka)
# Maintain output window until user presses a key
cv2.waitKey(0)
# Destroying present windows on screen
cv2.destroyAllWindows()
