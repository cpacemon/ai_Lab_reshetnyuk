import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Завантаження зображення
img = cv2.imread('coins_2.jpg')
cv2.imshow("Initial image", img)
cv2.waitKey()

# Перетворюємо зображення з BGR у RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Застосовуємо фільтр pyrMeanShift для спрощення кольорової палітри
filtro = cv2.pyrMeanShiftFiltering(img, 20, 40)

# Перетворюємо зображення у відтінки сірого
gray = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)

# Застосовуємл бінаризацію
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Image binarization", thresh)
cv2.waitKey()

# Знаходимо контури об'єктів
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Відображаємо контури на оригінальному зображенні
img_contours = img_rgb.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
cv2.imshow("Coin contours", img_contours)
cv2.waitKey()

# Створюємо маску
markers = np.zeros_like(gray, dtype=np.int32)

# Заповнюємо маску маркерами
for i, contour in enumerate(contours):
    cv2.drawContours(markers, [contour], 0, i + 1, -1)

# Застосовуємо дистанційну трансформацію
dist = ndi.distance_transform_edt(thresh)
cv2.imshow("Distance transform", dist)
cv2.waitKey()

# Знаходимо локальні максимуми в дистанційному перетворенні
local_max = peak_local_max(dist, min_distance=20, labels=markers)

# Використовуємо np.clip, щоб упевнитись, що значення індексів знаходяться в межах масиву
local_max[:, 0] = np.clip(local_max[:, 0], 0, markers.shape[0] - 1)
local_max[:, 1] = np.clip(local_max[:, 1], 0, markers.shape[1] - 1)

# Застосуємо маркери локальних максимумів до маски
markers[local_max[:, 0], local_max[:, 1]] = 0

# Застосуємо водорозділ
labels = watershed(-dist, markers, mask=thresh)

# Перетворимо labels на зображення з трьома каналами
labels_display = cv2.cvtColor(labels.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Створюємо кольорову маску
color_mask = np.zeros_like(img_rgb)
for i in range(1, labels.max() + 1):
    color_mask[labels == i] = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)

# Змішуємо кольорову маску з зображенням
result = cv2.addWeighted(img_rgb, 0.7, color_mask, 0.7, 0)

# Відображення фінального результату
cv2.imshow("Final result", result)
cv2.waitKey()
cv2.destroyAllWindows()

