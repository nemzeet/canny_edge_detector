import cv2 as cv
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape
from scipy import ndimage
from scipy.ndimage.measurements import median

# original RGB, Gray image
rgb_img = cv.cvtColor(cv.imread('img.jpg'), cv.COLOR_BGR2RGB)
gray_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2GRAY)

# grayscale image for using 
img = gray_img.copy()

# Stage 1
def myBlur(img):
    k = np.array([[2, 4, 5, 4, 2],
             [4, 9, 12, 9, 4],
             [5, 12, 15, 12, 5],
             [4, 9, 12, 9, 4],
             [2, 4, 5, 4, 2]])* (1/159)
    img = cv.filter2D(img, -1, k)

    return img

# Stage 2
def mySobelFilter(img ,low):
    
    # 8bit sobelX
    sobelx_64 = cv.Sobel(img,cv.CV_32F,1,0,ksize=3)
    absx_64 = np.absolute(sobelx_64)
    sobelx_8u1 = absx_64/absx_64.max()*255
    sobelx_8u = np.uint8(sobelx_8u1)
    
    # 8bit SobelY
    sobely_64 = cv.Sobel(img,cv.CV_32F,0,1,ksize=3)
    absy_64 = np.absolute(sobely_64)
    sobely_8u1 = absy_64/absy_64.max()*255
    sobely_8u = np.uint8(sobely_8u1)
    
    # 8bit magnitude
    mag = np.hypot(sobelx_8u, sobely_8u)
    mag = mag / mag.max()*255
    mag = np.uint8(mag)
    
    # direction, angle
    theta = np.arctan2(sobely_64, sobelx_64)
    angle = np.rad2deg(theta)
    
    
    # low Thresholding
    M, N = mag.shape
    zero = 0 # white
    
    zero_i, zero_j = np.where(mag < low)    
    mag[zero_i, zero_j] = zero
    
    
    return (mag, angle)
    
            

# Stage 3
def myEdgeThinning(G, angle):
    M, N = G.shape
    C = np.zeros((M, N), dtype = np.uint8) # 해당 픽셀이 검사 되었는지 확인
    
    # ------------------------------ 0, 45, 90, 135로 방향 조정
    for i in range(0, M):
        for j in range(0, N):
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
                    angle[i, j] = 0
                elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
                    angle[i, j] = 45
                elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
                   angle[i, j] = 90
                elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
                    angle[i, j] = 135
                    
    # ------------------------------ 동일 방향에서 max Gradient값 도출
    for i in range(0, M):
        for j in range(0, N):
            try:
                if G[i, j] != 0 and C[i, j] == 0:
                    kernel = []
                    gradients = []
                    cAngle = angle[i, j]
                    
                    for forward in [False, True]: 
                        a, b = i, j
                        while ((G[a, b] != 0 and (0<=a<=M and 0<=b<=N)) and angle[a,b] == cAngle):
                            kernel.append((a,b))
                            gradients.append(G[a,b])
                            
                            if cAngle == 0:
                                b = b + 1 if forward else b - 1
                            elif cAngle == 45:
                                a = a + 1 if forward else a - 1
                                b = b - 1 if forward else b + 1
                            elif cAngle == 90:
                                a = a + 1 if forward else a - 1
                            elif cAngle == 135:
                                a = a + 1 if forward else a - 1
                                b = b + 1 if forward else b - 1
                            
                            forward = True
                            
                    for k, l in kernel:
                        if G[k, l] ==  max(gradients):
                            C[k, l] = 1
                        elif G[k, l] <  max(gradients):
                            G[k, l] = 0
                            C[k, l] = 1
                                             
            except IndexError as e:
                pass
    
    return G
    

# Stage 4    
def myRealEdge(G, low, high):
    
    # ------------------------------ double thresholding (True Edge 검출)    
    weak = 40 # yellow
    strong = 255 # red
    
    strong_i, strong_j = np.where(G >= high)
    weak_i, weak_j = np.where((G < high) & (G >= low))
        
    G[strong_i, strong_j] = strong
    G[weak_i, weak_j] = weak


    # top->bottom, bottom->top, left->right, right->left 순으로 순회하며 true edge 검출    
    M, N = G.shape
    WG, RG = G.copy(), G.copy()
    start_r = (1, M-1, 1, M-1)
    start_c = (1, N-1, N-1, 1)
    end_r = (M,0,M,0)
    end_c = (N,0,0,N)
    step_r = (1,-1,1,-1)
    step_c = (1,-1,-1,1)
    
    for d in range(4):
        for i in range(start_r[d], end_r[d], step_r[d]):
            for j in range(start_c[d], end_c[d], step_c[d]):
                if (G[i,j] == weak):
                    try:
                        if ((G[i+1, j-1] == strong) or (G[i+1, j] == strong) or (G[i+1, j+1] == strong)
                            or (G[i, j-1] == strong) or (G[i, j+1] == strong)
                            or (G[i-1, j-1] == strong) or (G[i-1, j] == strong) or (G[i-1, j+1] == strong)):
                            G[i, j] = strong
                            WG[i, j] = 170 
                            RG[i, j] = 170 # blue
                        else:
                            WG[i, j] = weak
                            RG[i, j] = 0
                    except IndexError as e:
                        pass

    return WG, RG, strong
    

# Final Result
def checkingEdge(img1, img2, img3, strong):
    M, N = img2.shape
    
    for i in range(0, M):
        for j in range(0, N):
            try:
                if img2[i, j] == 0:
                    n = int(img3[i, j])
                    img1[i, j] = [n, n, n]
                else:
                    if img2[i, j] == strong:
                        img1[i, j] = [255, 0, 0]
                    elif img2[i, j] == 170: # true weak Edge
                        img1[i, j] = [0, 0, 255]
            except IndexError as e:
                pass
     
    return img1




# ------------------------------ create (hsv + white) colormap
white = np.ones((256, 4))
white[:, 0] = np.linspace(256 / 256, 1, 256)
white[:, 1] = np.linspace(256 / 256, 1, 256)
white[:, 2] = np.linspace(256 / 256, 1, 256)
white_cmp = ListedColormap(white)
color_cmp = plt.cm.get_cmap('hsv', 255)
newcolors = np.vstack((white_cmp(np.linspace(0, 1, 1)),
                       color_cmp(np.linspace(0, 1, 255))))
white_hsv = ListedColormap(newcolors, name='white_hsv')


# ------------------------------ plot_img
def plot_img(index, img, title, cmap):
    plt.subplot(2, 4, index), plt.axis('off'), plt.title(title)
    if cmap is None:
         plt.imshow(img)
    else:
        plt.imshow(img, cmap)

# ------------------------------ high, low value
low = 12
high = 31

# ------------------------------ Running 
plot_img(1, rgb_img, 'origin img', None)

plot_img(2, img, 'gray scale', 'gray')

img = myBlur(img)
plot_img(3, img, 'stage 1', 'gray')

mag, angle = mySobelFilter(img, low)
plot_img(4, mag, 'stage 2', 'gray')

mag = myEdgeThinning(mag, angle)
plot_img(5, mag, 'stage 3', 'gray')

mag1, mag2, strong = myRealEdge(mag, low, high)
plot_img(6, mag1, 'stage 4', white_hsv)
plot_img(7, mag2, 'Final Result', white_hsv)

img = checkingEdge(rgb_img, mag2, gray_img, strong)
plot_img(8, img, 'Check', None)


plt.subplots_adjust(0, 0, 1, 0.9, 0.1, 0.05)
plt.suptitle('My Canny Edge')
plt.show()