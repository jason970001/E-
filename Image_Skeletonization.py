import cv2
import numpy as np
import os

def skeletonize(img):
    # 確保輸入圖像為單通道
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 將圖像轉換為二值圖像（確保黑底白字）
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 獲取結構元素
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    done = False
    size = np.size(binary)
    skel = np.zeros(binary.shape, np.uint8)
    
    while not done:
        # 執行開運算
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        
        # 獲取骨架
        temp = cv2.subtract(binary, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary = eroded.copy()
        
        # 如果沒有白色像素，則完成
        zeros = size - cv2.countNonZero(binary)
        if zeros == size:
            done = True
    
    # 反轉顏色，使其變為白底黑字
    return cv2.bitwise_not(skel)

def process_right_half(img):
    # 轉換為灰度圖
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 獲取圖像的寬度和高度
    height, width = gray.shape
    
    # 對右半部分進行骨架化
    right_half = gray[:, width//2:]
    right_skeleton = skeletonize(right_half)
    
    # 創建一個白色背景的三通道圖像
    result = np.full((height, width, 3), 255, dtype=np.uint8)
    
    # 將左半部分轉換為白底黑字
    left_half = cv2.threshold(gray[:, :width//2], 127, 255, cv2.THRESH_BINARY)[1]
    result[:, :width//2] = cv2.cvtColor(left_half, cv2.COLOR_GRAY2BGR)
    
    # 將右半部分的骨架放入結果圖像
    result[:, width//2:] = cv2.cvtColor(right_skeleton, cv2.COLOR_GRAY2BGR)
    
    return result

def process_images(input_folder, output_folder):
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 獲取輸入資料夾中的所有圖片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for image_file in image_files:
        # 構建完整的輸入文件路徑
        input_path = os.path.join(input_folder, image_file)
        
        # 讀取圖像
        img = cv2.imread(input_path)
        
        if img is None:
            print(f"無法讀取圖像: {image_file}")
            continue
        
        print(f"處理圖像: {image_file}")
        print(f"  原始圖像形狀: {img.shape}")
        print(f"  原始圖像類型: {img.dtype}")
        
        # 處理圖像的右半部分
        processed_img = process_right_half(img)
        
        print(f"  處理後圖像形狀: {processed_img.shape}")
        print(f"  處理後圖像類型: {processed_img.dtype}")
        
        # 構建輸出文件路徑
        output_filename = f"half_skeleton_{image_file}"
        output_path = os.path.join(output_folder, output_filename)
        
        # 保存結果
        cv2.imwrite(output_path, processed_img)
        print(f"  已保存: {output_filename}")
        print("--------------------")

# 設置輸入和輸出資料夾
input_folder = r"C:\Users\lifelab\gan-v202305_gui\dir"
output_folder = r"C:\Users\lifelab\gan-v202305_gui\dir2"

# 處理圖像
process_images(input_folder, output_folder)

print("所有圖像處理完成。")