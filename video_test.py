import cv2 as cv
import numpy as np

vid = cv.VideoCapture('test_videos/challenge.mp4')

size = (int(vid.get(3)), int(vid.get(4)))

# result = cv.VideoWriter('test.mp4',
#                         cv.VideoWriter_fourcc(*'MJPG'),
#                         24, size)
while True:
    ret, frame = vid.read()

    if ret:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_copy = frame.copy()
        h = frame.shape[0]
        w = frame.shape[1]

        polygon = np.array([[[w, h], [int(w/2), int(h * 0.55)], [int(w - w * 0.8), h]]])
        mask = np.zeros_like(frame_gray)
        mask = cv.fillPoly(mask, polygon, 255)

        blur = cv.GaussianBlur(frame_gray, (5, 5), 0)
        edges = cv.Canny(blur, 100, 200)

        match = cv.bitwise_and(edges, mask)

        lines = cv.HoughLinesP(match, 4, np.pi / 180, 100, np.array([]), minLineLength=1, maxLineGap=50)
        img_lines = frame.copy()
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv.imshow('video', img_lines)
        # result.write(img_lines)
    else:
        vid.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    if cv.waitKey(10) == ord('q'):
        break

vid.release()
cv.destroyAllWindows()