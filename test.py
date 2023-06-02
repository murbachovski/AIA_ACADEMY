import cv2
import numpy as np

# 이미지를 로드합니다.
image = cv2.imread('dog.402.jpg')

# 지우고자 하는 영역의 좌표와 크기를 지정합니다.
x, y, width, height = 100, 87, 29, 53

# 관심 영역(ROI)을 추출합니다.
roi = image[y:y+height, x:x+width]

# 마스크를 생성하여 관심 영역을 지정합니다.
mask = np.zeros(image.shape[:2], dtype=np.uint8)
mask[y:y+height, x:x+width] = 255

# 마스크를 사용하여 관심 영역을 지웁니다.
result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

# 결과 이미지를 출력합니다.
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
