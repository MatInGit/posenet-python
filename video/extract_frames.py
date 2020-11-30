import cv2
vidcap = cv2.VideoCapture('videoplayback.mp4')
success,image = vidcap.read()
print(len(image))
count = 0
cnt = 0
l_lim = 1500
u_lim =  l_lim+3000

while success:
    #print(count)
    if (count % 20 == 0) and (count >= l_lim) and (count <= u_lim):
        print(count)
        cv2.imwrite("frames/frame%d.jpg" % cnt, image)     # save frame as JPEG file
        cnt+=1
        print('Read a new frame: ', success)
    count += 1
    success,image = vidcap.read()
    if (count >= u_lim):
        break
