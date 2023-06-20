import cv2 as cv
import numpy as np
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
import time
import math
from numba import jit

form_class = uic.loadUiType("./detect.ui")[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('x-ray defect detection program')
        
        # 열기 버튼 클릭 이벤트
        self.openBtn.clicked.connect(self.fileOpenFunction)
                
        # 검출 버튼 클릭 이벤트
        self.detectBtn.clicked.connect(self.detectFunction)
        
        # 라디오버튼 클릭 이벤트
        self.bananaBtn.clicked.connect(self.bananaFunction)
        self.carrotBtn.clicked.connect(self.carrotFunction)
        self.onionBtn.clicked.connect(self.onionFunction)
        self.orangeBtn.clicked.connect(self.orangeFunction)
        self.pimentoBtn.clicked.connect(self.pimentoFunction)
        self.sweet_potatoBtn.clicked.connect(self.sweet_potatoFunction)


    def fileOpenFunction(self):
        # 이미지 개수, pixmap1 초기화
        self.step = 0
        self.atextLabel.setText(' ')
        pixmap1 = QPixmap()
        self.afterLabel.setPixmap(pixmap1)
        
        global pseudo_img, path
        pseudo_img=[]
        path=[]
        
        global fname
        fname = QFileDialog.getOpenFileNames(self, 'file')
        
        self.img = cv.imread(fname[0][0], 0)
        if self.img is None: sys.exit('파일을 찾을 수 없습니다')
        
        # 이미지 설정
        pixmap = QPixmap(fname[0][0]).scaled(945, 738, Qt.KeepAspectRatio)
        self.beforeLabel.setPixmap(pixmap)
        self.beforeLabel.resize(pixmap.width(), pixmap.height())
        
        # 이미지 하단에 텍스트 설정
        self.btextLabel.setText(str(self.step+1)+"/"+str(len(fname[0])))
        
        
    def detectFunction(self):        
        for i in range(len(fname[0])):
            img = cv.imread(fname[0][i], 0)
            
            bimg_tmp = convert_binary_oppening(img, 240, 5, 5) # 침식 5번, 팽창 5번
            info_tmp = np.where(bimg_tmp==0, 1, 0)
            
            bimg = convert_binary_oppening(img, 240, 5, 2) # 침식 5번, 팽창 2번
            info = np.where(bimg==0, 1, 0)
            
            basic_img = np.where(info_tmp==1, img, 255)
            simg = stretching(basic_img, info_tmp)
            
            start = time.time()
            pseudo_img.append(defect_detection(simg, info, flag, radius, 16.0))
            end = time.time()
            print(f"{end - start:.5f} sec")
            path.append('C:/Users/osoyo/.spyder-py3/capstone/'+str(i)+'.tif')
            cv.imwrite(path[i], cv.cvtColor(pseudo_img[i], cv.COLOR_RGB2BGR))
            
        # 이미지 설정
        pixmap1 = QPixmap(path[self.step]).scaled(945, 738, Qt.KeepAspectRatio)
        
        self.afterLabel.setPixmap(pixmap1)
        self.afterLabel.resize(pixmap1.width(), pixmap1.height())
        self.atextLabel.setText(str(self.step+1)+"/"+str(len(fname[0])))
      
        
    def bananaFunction(self):
        global flag, radius, pnum
        flag = 0.6; radius = 0.0; pnum = 50
        
    def carrotFunction(self):
        global flag, radius, pnum
        flag = 0.0; radius = 0.0; pnum = 50
    
    def onionFunction(self):
        global flag, radius, pnum
        flag = 0.0; radius = 20.0; pnum = 20
    
    def orangeFunction(self):
        global flag, radius, pnum
        flag = 0.0; radius = 20.0; pnum = 50
    
    def pimentoFunction(self):
        global flag, radius, pnum
        flag = 1.85; radius = 0.0; pnum = 45
        
    def sweet_potatoFunction(self):
        global flag, radius, pnum
        flag = 0.9; radius = 0.0; pnum = 50
        
        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_A and self.step != 0:
            self.step -= 1
            
        elif e.key() == Qt.Key_D and self.step != len(fname[0])-1:
            self.step += 1
        
        pixmap = QPixmap(fname[0][self.step]).scaled(945, 738, Qt.KeepAspectRatio)
        self.beforeLabel.setPixmap(pixmap)
        self.btextLabel.setText(str(self.step+1)+"/"+str(len(fname[0])))
        
        pixmap1 = QPixmap(path[self.step]).scaled(945, 738, Qt.KeepAspectRatio)
        self.afterLabel.setPixmap(pixmap1)
        self.atextLabel.setText(str(self.step+1)+"/"+str(len(fname[0])))
        
        
def convert_binary_oppening(img, threshold, num1, num2):
    se = np.uint8([[0,0,1,0,0],  # shape
                   [0,1,1,1,0],
                   [1,1,1,1,1],
                   [0,1,1,1,0],
                   [0,0,1,0,0]])
    
    ret = np.where(img >= threshold, 255, 0).astype(np.uint16)
    ret = cv.dilate(ret, se, iterations=num1) # 침식 (검은 픽셀 기준이므로)
    ret = cv.erode(ret, se, iterations=num2) # 팽창 (검은 픽셀 기준이므로)
    
    return ret


def stretching(img, info):
    low = np.min(img)
    high = np.max(np.where(info==1, img, 0))
    
    def func(x):
        if x == 255:
            return 255
        return np.uint8((x-low)/(high-low) * 255)
    
    vfunc = np.vectorize(func)
    ret = vfunc(img)
    return ret


@jit(nopython=True)
def cal_center(img):
    h, w = img.shape
    
    sum_y = 0
    sum_x = 0
    num = 0
    
    for i in range(h):
        for j in range(w):
            if img[i, j] != 255:
                sum_y += i
                sum_x += j
                num += 1
                
    mid_y = sum_y // num
    mid_x = sum_x // num
    
    return mid_y, mid_x

        
@jit(nopython=True)
def dfs(y, x, visited, cur_size, y_min, x_min, y_max, x_max, dy, dx, h, w, defect_pixels_matrix):
    if visited[y][x]:
        return int(1e5), int(1e5), -1, -1, 0

    visited[y][x] = True
    
    y_min = y_max = y
    x_min = x_max = x
    cur_size = 1

    for i in range(4):
        ny = y + dy[i]
        nx = x + dx[i]
        adj_size = 0
        if 0 <= ny < h and 0 <= nx < w and defect_pixels_matrix[ny][nx]:
            ny_min, nx_min, ny_max, nx_max, n_size = dfs(ny, nx, visited, cur_size, y_min, x_min, y_max, x_max, dy, dx, h, w, defect_pixels_matrix)
            cur_size += n_size
            y_min = min(y_min, ny_min)
            y_max = max(y_max, ny_max)
            x_min = min(x_min, nx_min)
            x_max = max(x_max, nx_max)

    return y_min, x_min, y_max, x_max, cur_size


@jit(nopython=True)
def get_defect_candidates(pseudo_img, th1=0.9, th2=0.8):
    pixel_range = np.max(pseudo_img)
    h, w, _ = pseudo_img.shape
    
    num = 0
    defect_pixels = [[-1, -1] for _ in range(100000)]
    defect_pixels_matrix = [[False for j in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if pseudo_img[i][j][0] > pixel_range * th1 and pseudo_img[i][j][1] < pixel_range * th2:
                defect_pixels[num] = [i, j]
                defect_pixels_matrix[i][j] = True
                num += 1
    
    dy = [-1, 0, 1, 0]
    dx = [0, 1, 0, -1]
    visited = [[False for j in range(w)] for _ in range(h)]
    candidates_info = [[0, 0, 0, 0, 0] for _ in range(10000)]
    
    cur_size = 0
    y_min = x_min = int(1e5)
    y_max = x_max = -1
    
    candidates_num = 0
    for i, j in defect_pixels[:num]:
        if visited[i][j]:
            continue
        y_min, x_min, y_max, x_max, cur_size = dfs(i, j, visited, cur_size, y_min, x_min, y_max, x_max, dy ,dx, h, w, defect_pixels_matrix)
        candidates_info[candidates_num] = [y_min, x_min, y_max, x_max, cur_size]
        candidates_num += 1
        
        cur_size = 0
        y_min = x_min = int(1e5)
        y_max = x_max = -1
    
    ret = candidates_info[:candidates_num]
    ret.sort(key = lambda x : -x[4])
    
    return ret
    

@jit(nopython=True)
def defect_detection(img, info, flag=0.0, radius=0.0, option=1.0):
    num = 0
    height, width = img.shape

    alpha = 100 # 색상 선택을 위한 용도
    scale = 20 # n x n WAM 필터의 크기
    output = [[[255, 255, 255] for j in range(width)] for i in range(height)]
    
    pss = [[0]*(width+1) for _ in range(height+1)]
    ps = [[0]*(width+1) for _ in range(height+1)]
    
    for i in range(1, height+1):
        for j in range(1, width+1):
            num += 1
            ps[i][j] = ps[i-1][j] + ps[i][j-1] - ps[i-1][j-1] + img[i-1, j-1]
            pss[i][j] = pss[i-1][j] + pss[i][j-1] - pss[i-1][j-1] + np.int64(img[i-1, j-1])*img[i-1, j-1]
        
    if radius != 0.0:
        mid_y, mid_x = cal_center(img)
    
    p = 5
    q = 10
    
    # RD 이미지 생성
    for r in range(20, height - 20):
        for c in range(20, width - 20):
            num += 1
            i = r + 1;
            j = c + 1;

            # 한 픽셀에 대해 주변 40*40 범위의 평균과 표준편차를 구해 주변보다 밀도가 얼마나 차이나는지 구함
            try:
                if info[r, c] == 1 and img[r, c] < 200:
                    wam = (ps[i + scale][j + scale] - ps[i - scale-1][j + scale] - ps[i + scale][j - scale-1] + ps[i - scale-1][j - scale-1]) / (41 * 41)
                    lam = (ps[i+2][j+2] - ps[i-2-1][j+2] - ps[i+2][j-2-1] + ps[i-2-1][j-2-1]) / (5 * 5)
                    
                    sq_sum = pss[i + scale][j + scale] - pss[i - scale-1][j + scale] - pss[i + scale][j - scale-1] + pss[i - scale-1][j - scale-1]
                    sq_sum_avg = sq_sum / (41 * 41)
                    std = math.sqrt(sq_sum_avg - wam*wam)

                    z = round((lam - wam) / std, 2)
                    
                    if info[r, c] == 1:
                        C = (ps[i+1][j+1] - ps[i-1-1][j+1] - ps[i+1][j-1-1] + ps[i-1-1][j-1-1]) / (3 * 3)
                        
                        # row
                        A = (ps[i][j+p] - ps[i][j-p-1] - ps[i-q-1][j+p] + ps[i-q-1][j-p-1]) / ((2*p+1) * (q+1))
                        B = (ps[i+q][j+p] - ps[i+q][j-p-1] - ps[i-1][j+p] + ps[i-1][j-p-1]) / ((2*p+1) * (q+1))
                        A_std = math.sqrt((pss[i][j+p] - pss[i][j-p-1] - pss[i-q-1][j+p] + pss[i-q-1][j-p-1]) / ((2*p+1) * (q+1)) - A**2)
                        B_std = math.sqrt((pss[i+q][j+p] - pss[i+q][j-p-1] - pss[i-1][j+p] + pss[i-1][j-p-1]) / ((2*p+1) * (q+1)) - B**2)
                        
                        if abs(C-A) < abs(C-B):
                            val = round((A-B) / A_std, 2)
                        else:
                            val = round((B-A) / B_std, 2)
                        
                        # column
                        A = (ps[i+p][j] - ps[i-p-1][j] - ps[i+p][j-q-1] + ps[i-p-1][j-q-1]) / ((2*p+1) * (q+1))
                        B = (ps[i+p][j+q] - ps[i-p-1][j+q] - ps[i+p][j-1] + ps[i-p-1][j-1]) / ((2*p+1) * (q+1))
                        A_std = math.sqrt((pss[i+p][j] - pss[i-p-1][j] - pss[i+p][j-q-1] + pss[i-p-1][j-q-1]) / ((2*p+1) * (q+1)) - A**2)
                        B_std = math.sqrt((pss[i+p][j+q] - pss[i-p-1][j+q] - pss[i+p][j-1] + pss[i-p-1][j-1]) / ((2*p+1) * (q+1)) - B**2)
                        
                        if abs(C-A) < abs(C-B):
                            val += round((A-B) / A_std, 2)
                        else:
                            val += round((B-A) / B_std, 2)
                        
                        if val > 0:
                            z += val / option
                    
                    if 0 < z < flag:
                        z *= -1.0
                        
                    # orange, onion
                    if radius > 0.0:
                        dist = math.sqrt((mid_y-r)**2 + (mid_x-c)**2)
                        if dist <= radius:
                            z = -abs(z)    
                    
                    if z > 0:  # Red
                        v = max(0, 255 - int(alpha * z))
                        output[r][c][0] = 255  # R
                        output[r][c][1] = v  # G
                        output[r][c][2] = v  # B
                    else: # Blue
                        v = max(0, 255 + int(alpha * z))
                        output[r][c][0] = v  # R
                        output[r][c][1] = v  # G
                        output[r][c][2] = 255  # B
            except:
                pass
    output = np.array(output, dtype=np.uint8)
    
    ret = output
    defect_candidates = get_defect_candidates(ret, 0.9, 0.8)

    # 색상 및 굵기
    p = (50, 100, 0) # RGB, Box Colour
    lt = 5 # line thickness
    for y_min, x_min, y_max, x_max, pixel_num in defect_candidates:
        if pixel_num <= pnum: 
            continue
        
        if y_max - y_min <= 3 or x_max - x_min <= 3:
            continue
        
        y_min -= 10
        x_min -= 10
        y_max += 10
        x_max += 10
    
        # -
        ret[y_min:y_min+lt, x_min:x_max+lt, 0] = p[0]
        ret[y_min:y_min+lt, x_min:x_max+lt, 1] = p[1]
        ret[y_min:y_min+lt, x_min:x_max+lt, 2] = p[2]
        
        # _
        ret[y_max:y_max+lt, x_min:x_max+lt, 0] = p[0]
        ret[y_max:y_max+lt, x_min:x_max+lt, 1] = p[1]
        ret[y_max:y_max+lt, x_min:x_max+lt, 2] = p[2]
        
        # |
        ret[y_min:y_max+lt, x_min:x_min+lt, 0] = p[0]
        ret[y_min:y_max+lt, x_min:x_min+lt, 1] = p[1]
        ret[y_min:y_max+lt, x_min:x_min+lt, 2] = p[2]
        
        #   |
        ret[y_min:y_max+lt, x_max:x_max+lt, 0] = p[0]
        ret[y_min:y_max+lt, x_max:x_max+lt, 1] = p[1]
        ret[y_min:y_max+lt, x_max:x_max+lt, 2] = p[2]
        
        break
        
    return ret


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = WindowClass()
    win.showMaximized()
    sys.exit(app.exec_())
