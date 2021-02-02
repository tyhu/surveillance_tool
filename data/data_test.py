import cv2

cap = cv2.VideoCapture('PRG7.avi')
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None


count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret: break
    if out is None:
        w,h = frame.shape[0], frame.shape[1]
        out = cv2.VideoWriter('output.avi',fourcc, fps, (h,w))
    count+=1
    if count>140*fps and count<184*fps:
        out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
