import sys
import cv2

img_path = sys.argv[1]

img = cv2.imread(img_path)
new_path = img_path[:-3] + "png"
print(new_path)
cv2.imwrite(new_path, img)
