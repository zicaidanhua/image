import cv2
import numpy as np
import time

origin_image1 = cv2.imread('1.jpg')
origin_image2 = cv2.imread('2.jpg')
# cv2.imshow("1",origin_image1)
# cv2.imshow("2",origin_image2)
# cv2.waitKey(0)

sift = cv2.SIFT_create()
#
# #获取各个图像的特征点以及sift特征向量
# #返回
(kp1, des1) = sift.detectAndCompute(origin_image1, None)
(kp2, des2) = sift.detectAndCompute(origin_image2, None)
# # print("=============================")
# # print(kp)
# # print("=============================")
# # print(des)
# print("=============================")
# print(des1.shape[0])#特征点的数目
# print("=============================")
# print(des2.shape[0])#特征点的数目
#
# #举例说明kp中的参数信息
# # for i in range(2):
# #     print("关键点", i)
# #     print("数据类型:", type(kp[i]))
# #     print("关键点坐标:", kp[i].pt)
# #     print("邻域直径:", kp[i].size)
# #     print("方向:", kp[i].angle)
# #     print("所在的图像金字塔的组:", kp[i].octave)
#

#绘制特征点
# sift_origin1 = cv2.drawKeypoints(origin_image1, kp1, origin_image1, color=(255, 0, 255))
# # cv2.imshow("1",sift_origin1)
# sift_origin2 = cv2.drawKeypoints(origin_image2, kp2, origin_image2, color=(255, 0, 255))
# # cv2.imshow("2",sift_origin2)
# # cv2.waitKey(0)
# sift_origin2 = sift_origin2[0:len(sift_origin1)]
#
# sift_cat1 = np.hstack((sift_origin1, sift_origin2))#对于提取特征点后的图像进行横向的拼接,特征点的大小需要相同
# cv2.imwrite("sift_cat1.jpg", sift_cat1)
# cv2.imshow("sift_point",sift_cat1)
# cv2.waitKey()


#特征点匹配
#K近邻算法求取空间距离最近的K个数据点，将数据点归为一类
start = time.time()#计算匹配点和匹配时间
bf = cv2.BFMatcher()#Brute-Force蛮力匹配BF匹配器使用cv2.BFMatcher()创建BFMetcher对象需要两个可选参数1，normtype指定要测量的距离，crossCheck
matches1 = bf.knnMatch(des1, des2, k=2)#BFMatcher.match()返回最佳匹配，knnMatch()返回k个最佳匹配
# 使用cv2.drawMatches()来绘制匹配的点，它会将两幅图像先水平排列，然后在最佳匹配的点之间绘制直线。
# 如果前面使用的BFMatcher.knnMatch()，现在可以使用函数cv2.drawMatchsKnn为每个关键点和它的个最佳匹配点绘制匹配线。

ratio1 = 0.01
good1 = []

for m1, n1 in matches1:
    #如果最接近和此接近的比值大于一个既定的值，那么可以保留最接近的值，认为匹配的点为good_match
    if m1.distance < ratio1 * n1.distance:
        good1.append([m1])
end = time.time()
# print("匹配点匹配运行时间",(end-start))

#通过对于good值进行索引，可以指定特征点的数目进行匹配
match_result1 = cv2.drawMatchesKnn(origin_image1, kp1, origin_image2, kp2, good1, None, flags=2)
cv2.imwrite("match.jpg", match_result1)

#使用单应矩阵将匹配的图像通过选装，变换等方式与目标图像对齐
#单应性矩阵有8个参数，需要8个方程只需要4个反差


if len(good1) > 4:
    ptsA = np.float32([kp1[m[0].queryIdx].pt for m in good1]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good1]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    # RANSAC算法选择其中最优的四个点
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)#findHomography:计算多个二维点对之间的最优单映射变换矩阵
    #源平面中点的坐标矩阵，目标平面中点的坐标矩阵
    #0——利用所有点的常规方法 RANSAC基于RANSAC的鲁棒算法 LMEDS最下中值鲁棒算法 RHO 基于RHOSAC的鲁棒算法
    #ransacReprojThreshold将点对视为内点的最大允许冲投影错误阈值RANSAC和RHO方法
    #mask可选输出掩码矩阵，通常由(RANSAC或LMEDS)设置
    #maxltersRANSAC算法的最大迭代次数，默认值为2000
    #confidnce，可信度取值范围为0~1
    #计算出单应矩阵
    # print(status)
    imgout = cv2.warpPerspective(origin_image2, H, (origin_image1.shape[1]+origin_image2.shape[1], origin_image1.shape[0]),
                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    imgout[0:origin_image1.shape[0],0:origin_image1.shape[1]] = origin_image1
    #src:输入的图像
    #变化矩阵
    #dsize:输出图片的大小
    #flags插值方法
    #INTER_AREA/INRTER_BITS/INTER_BITS/INTER_CUNIC/INTER_LANCZOS4/INTER_LINAR/WARP_INVERSE_MAP
    #borderMode:边界模式
    #borderValue，边界模式为常量时的边界填充
    cv2.imwrite("imgout.png", imgout)
    # cv2.imshow("2", imgout)
    # cv2.waitKey()

image = cv2.imread('imgout.png')
img = cv2.medianBlur(image, 5)#中值滤波
b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
binary_image = b[1]
binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
indexes = np.where(binary_image == 255)  # 提取白色像素点的坐标
left = min(indexes[0])  # 左边界
right = max(indexes[0])  # 右边界
width = right - left  # 宽度
bottom = min(indexes[1])  # 底部
top = max(indexes[1])  # 顶部
height = top - bottom  # 高度
pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
cv2.imshow('picture', pre1_picture)
cv2.waitKey()
