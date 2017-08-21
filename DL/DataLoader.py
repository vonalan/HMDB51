import numpy as np

def read_data_from_video(video_path, num_frames = 10, new_size = (320,240)): 
    import cv2 
    
    frame_list = list()
    
    # open the video 
    # get informations about video: length, height, width, fps
    cap = cv2.VideoCapture(video_path)
    
    while(cap.isOpened()):
        ret, frame = cap.read() # RGB, three channels

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray, single channel 
            gray = cv2.resize(gray, new_size, interpolation = cv2.INTER_CUBIC) # resize
            frame_list.append(gray)
            # print(gray.shape)
            # cv2.imshow('gray',gray)
        else: 
            break 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # print(len(frame_list))
    import random
    index = random.sample([i for i in range(len(frame_list))], num_frames)
    frame_list = [frame_list[i] for i in range(len(frame_list)) if i in index]
    # print(len(frame_list))

    return frame_list

def list_to_matrix(frame_list): 
    mat = np.zeros((0,1))
    for frame in frame_list:
        mat = np.vstack((mat, frame.reshape((-1,1))))
    mat = mat/255.0 # important to normalize! 
    
    return mat.T 

def load(): 
    frame_list = read_data_from_video('vtest.avi', num_frames = 10, new_size = (80,60))
    frame_mat = list_to_matrix(frame_list)
    
    return frame_mat



if __name__ == '__main__': 
    frame_list = read_data_from_video('vtest.avi', num_frames = 10, new_size = (80,60))
    frame_mat = list_to_matrix(frame_list)
    print(frame_mat.shape)
    print(frame_mat.tolist())
    # np.savetxt("vtest.txt", frame_mat)