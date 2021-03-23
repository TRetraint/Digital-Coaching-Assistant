import cv2

cap = cv2.VideoCapture('video5.mp4')
w = int(cap.get(3))
h = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('video5Flip.mp4', 0x7634706d,30.0,(int(w),int(h)))

while True:
    ret, frame = cap.read()
    if ret:
        res = cv2.flip(frame, 1)
        out.write(res)
    else:
        break

cv2.destroyAllWindows()