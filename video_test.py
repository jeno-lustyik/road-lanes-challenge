import cv2 as cv
import numpy as np

vid = cv.VideoCapture('test_videos/solidWhiteRight.mp4')

# setting up a result file to save the video into.
size = (int(vid.get(3)), int(vid.get(4)))

result = cv.VideoWriter('test.mp4',
                        cv.VideoWriter_fourcc(*'MP4V'),
                        24, size)

# Loading the video
while True:
    ret, frame = vid.read()
    # making sure that the video is running if the value ret is not empty
    if ret:

        # Creating a grayscale frame and a copy of the frame
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_copy = frame.copy()

        # Taking the sizes of the video to create the polygon mask based on it.
        h = frame.shape[0]
        w = frame.shape[1]

        # Creating the polygon mask
        polygon2 = np.array([[[w, h], [int(w / 2 + w * 0.1), int(h * 0.6)], [int(w / 2 - w * 0.1), int(h * 0.6)], [int(w - w * 0.85), h]]])
        mask = np.zeros_like(frame_gray)
        mask = cv.fillPoly(mask, polygon2, 255)

        # Blurring, Canny edges
        blur = cv.GaussianBlur(frame_gray, (5, 5), 0)
        edges = cv.Canny(blur, 100, 200)

        # Bitwise op the edges and the mask together
        match = cv.bitwise_and(edges, mask)

        # Extract the lines from the bitwise, and draw them on the frame.
        lines = cv.HoughLinesP(match, 2, np.pi / 180, 100, np.array([]), minLineLength=1, maxLineGap=100)
        img_lines = frame.copy()
        if lines is not None:
            for i in range(len(lines)):
                for x1, y1, x2, y2 in lines[i]:
                    cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv.imshow('video', img_lines)
        result.write(img_lines) # line to saving the frames to the result file

    # If the video reaches its last frame, set it back to frame 0.
    else:
        vid.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    # Escape key.
    if cv.waitKey(10) == ord('q'):
        break

vid.release()
cv.destroyAllWindows()