import cv2

cap = cv2.VideoCapture('pull_ups.mp4')
w = int(cap.get(3))
h = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('pull_upsDiv4.mp4', 0x7634706d,30.0,(int(w/4),int(h/4)))

while True:
    ret, frame = cap.read()
    if ret:
        res = cv2.resize(frame, (int(w/4),int(h/4)))
        out.write(res)
    else:
        break

cv2.destroyAllWindows() 