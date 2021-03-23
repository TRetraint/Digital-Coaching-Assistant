import cv2

cap = cv2.VideoCapture('IMG_6606.mp4')
w = int(cap.get(3))
h = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('IMG_6606Div2.mp4', 0x7634706d,30.0,(int(w/2),int(h/2)))

while True:
    ret, frame = cap.read()
    if ret:
        res = cv2.resize(frame, (int(w/2),int(h/2)))
        out.write(res)
    else:
        break

cv2.destroyAllWindows() 