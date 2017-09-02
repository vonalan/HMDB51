import numpy as np 
import cv2 

def read_data_from_image(impath): 
    frame = cv2.imread(impath)
    # print(frame.shape)
    frame = cv2.resize(frame, (108,192), interpolation=cv2.INTER_CUBIC)
    # print(frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(frame.shape)

    cv2.imshow('IMAGE', frame)

    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

    return frame


# 定义鼠标事件回调函数
# def on_mouse(event, x, y, flags, param):
    
#     # 鼠标左键按下，抬起，双击
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print('Left button down at ({}, {})'.format(x, y))
#     elif event == cv2.EVENT_LBUTTONUP:
#         print('Left button up at ({}, {})'.format(x, y))
#     elif event == cv2.EVENT_LBUTTONDBLCLK:
#         print('Left button double clicked at ({}, {})'.format(x, y))

#     # 鼠标右键按下，抬起，双击
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         print('Right button down at ({}, {})'.format(x, y))
#     elif event == cv2.EVENT_RBUTTONUP:
#         print('Right button up at ({}, {})'.format(x, y))
#     elif event == cv2.EVENT_RBUTTONDBLCLK:
#         print('Right button double clicked at ({}, {})'.format(x, y))

#     # 鼠标中/滚轮键（如果有的话）按下，抬起，双击
#     elif event == cv2.EVENT_MBUTTONDOWN:
#         print('Middle button down at ({}, {})'.format(x, y))
#     elif event == cv2.EVENT_MBUTTONUP:
#         print('Middle button up at ({}, {})'.format(x, y))
#     elif event == cv2.EVENT_MBUTTONDBLCLK:
#         print('Middle button double clicked at ({}, {})'.format(x, y))

#     # 鼠标移动
#     elif event == cv2.EVENT_MOUSEMOVE:
#         print('Moving at ({}, {})'.format(x, y))

# # 为指定的窗口绑定自定义的回调函数
# cv2.namedWindow('Honeymoon Island')
# cv2.setMouseCallback('Honeymoon Island', on_mouse)

# frame = cv2.imread('imgtest.png')
# while True: 
#     cv2.imshow('Honeymoon Island', frame)
    
#     key = cv2.waitKey(0)

#     if key == 27: 
#         break
# cv2.destroyAllWindows()


if __name__ == '__main__': 
    read_data_from_image('imgtest.png')