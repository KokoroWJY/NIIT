#test2-3.py：捕获摄像头视频
import cv2

#创建窗口
cv2.namedWindow('video',cv2.WINDOW_NORMAL)
cv2.resizeWindow('video',640,480)
      		               
#创建VideoCapture对象，视频源为默认摄像头
vc=cv2.VideoCapture('D:\\Program Files (x86)\\Tencent\\QQ\\Tencent Files\\Megamind.avi')

#读取视频帧速率
fps= vc.get(cv2.CAP_PROP_FPS)     
print("帧速率：", fps)
#读取视频大小
size=(int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))) 	
print("大小：", size)


#循环读视频帧，直到视频结束  

size=(int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('daxiao: ', size)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vw=cv2.VideoWriter('Megamind.avi',fourcc,fps,size)

success,frame=vc.read()
#判断摄像头是否打开
flag = vc.isOpened()
print("摄像头是否打开", flag)

while success: 
    cv2.imshow('video',frame)   #将视频帧在窗口中显示

    vw.write(frame)

    success,frame=vc.read() #读下一帧        
    
    key=cv2.waitKey(1) #实时采集用1ms没问题
    
    if key==27:  #按Esc键结束                       
        break

#关闭视频
vc.release()                            
cv2.destroyAllWindows() 

