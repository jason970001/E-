# -*- coding: utf-8 -*-
import argparse
import os
import glob
import shutil
import _pickle as pickle
import numpy as np
import torch
from data import DatasetFromObj
from torch.utils.data import DataLoader, TensorDataset
from model import Zi2ZiModel
import random
import time
import math
from PIL import Image, ImageDraw, ImageFont, ImageTk
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from model.model import chk_mkdir
import cv2
import tkinter as tk
from tkinter import filedialog
import subprocess
import tkinter as tk
from tkinter import filedialog

writer_dict = {
        '智永': 0, ' 隸書-趙之謙': 1, '張即之': 2, '張猛龍碑': 3, '柳公權': 4, '標楷體-手寫': 5, '歐陽詢-九成宮': 6,
        '歐陽詢-皇甫誕': 7, '沈尹默': 8, '美工-崩雲體': 9, '美工-瘦顏體': 10, '虞世南': 11, '行書-傅山': 12, '行書-王壯為': 13,
        '行書-王鐸': 14, '行書-米芾': 15, '行書-趙孟頫': 16, '行書-鄭板橋': 17, '行書-集字聖教序': 18, '褚遂良': 19, '趙之謙': 20,
        '趙孟頫三門記體': 21, '隸書-伊秉綬': 22, '隸書-何紹基': 23, '隸書-鄧石如': 24, '隸書-金農': 25,  '顏真卿-顏勤禮碑': 26,
        '顏真卿多寶塔體': 27, '魏碑': 28
    }


parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('--experiment_dir', default='./',
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--start_from', type=int, default=0)
parser.add_argument('--gpu_ids', default=['cuda:0'], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--batch_size', type=int, default=1, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=161850, help='resume from previous training')
parser.add_argument('--obj_path', type=str, default='./obj/val.obj', help='the obj file you infer')
parser.add_argument('--input_nc', type=int, default=3)

parser.add_argument('--from_txt', action='store_true')
parser.add_argument('--src_txt', type=str, default='大威天龍大羅法咒世尊地藏波若諸佛')
parser.add_argument('--canvas_size', type=int, default=256)
parser.add_argument('--char_size', type=int, default=256)
parser.add_argument('--run_all_label', action='store_true')
parser.add_argument('--label', type=int, default=0)
parser.add_argument('--src_font', type=str, default='charset/gbk/方正新楷体_GBK(完整).TTF')
parser.add_argument('--type_file', type=str, default='type/宋黑类字符集.txt')

parser.add_argument('--input_dir', type=str, default='./input/', help='picture directory.')
parser.add_argument('--save_dir', type=str, default='./output/', help='directory for saving generated images.')
parser.add_argument('--input_path', type=str, default=None, help='picture directory')

args = parser.parse_args()

def draw_single_char(ch, font, canvas_size):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, (0, 0, 0), font=font)
    img = img.convert('L')
    return img


    val_path = os.path.join(package_obj_savepath, "val.obj")
    pickle_examples(sorted(glob.glob(os.path.join(package_pic_dir, "*.jpg"))), val_path=val_path)
def main():
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    chk_mkdir(infer_dir)

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    t0 = time.time()

    model = Zi2ZiModel(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    t1 = time.time()

    if args.from_txt:
        src = args.src_txt
        font = ImageFont.truetype(args.src_font, size=args.char_size)
        img_list = [transforms.Normalize(0.5, 0.5)(
            transforms.ToTensor()(
                draw_single_char(ch, font, args.canvas_size)
            )
        ).unsqueeze(dim=0) for ch in src]
        label_list = [args.label for _ in src]

        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor(label_list)

        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    else:
        """
        val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'),
                                     input_nc=args.input_nc,
                                     start_from=args.start_from)
        """
        val_dataset = DatasetFromObj(os.path.join(args.obj_path),
                                     input_nc=args.input_nc,
                                     start_from=args.start_from)
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    global_steps = 0
    with open(args.type_file, 'r', encoding='utf-8') as fp:
        fonts = [s.strip() for s in fp.readlines()]
    writer_dict = {v: k for k, v in enumerate(fonts)}

    for batch in dataloader:
        if args.run_all_label:
            # global writer_dict
            writer_dict_inv = {v: k for k, v in writer_dict.items()}
            for label_idx in range(29):
                model.set_input(torch.ones_like(batch[0]) * label_idx, batch[2], batch[1])
                model.forward()
                tensor_to_plot = torch.cat([model.fake_B, model.real_B], 3)
                # img = vutils.make_grid(tensor_to_plot)
                save_image(tensor_to_plot, os.path.join(infer_dir, "infer_{}".format(writer_dict_inv[label_idx]) + "_construct.png"))
        else:
            # model.set_input(batch[0], batch[2], batch[1])
            # model.optimize_parameters()
            model.sample(batch, infer_dir)
            global_steps += 1

    t_finish = time.time()

    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))

#尋找contour列表中每一個元素的角落座標的函式
def find_contour_corner(contours):
    left = contours[0][0][0][0]
    right = contours[0][0][0][0]
    top = contours[0][0][0][1]
    bottom = contours[0][0][0][1]
    
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            if contours[i][j][0][0] < left:
                left = contours[i][j][0][0]
            if contours[i][j][0][0] > right:
                right = contours[i][j][0][0]
            if contours[i][j][0][1] < top:
                top = contours[i][j][0][1]
            if contours[i][j][0][1] > bottom:
                bottom = contours[i][j][0][1]
    
    return left, right, top, bottom



#將手寫文字大小調整並儲存的函式
def resize_fontsize(filename):
    global save_crop_path
    #讀取檔案
    img = cv2.imread(filename)
    img_width , img_height = img.shape[:2]
    #轉成灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #二值化
    ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
    
    #找輪廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #找出文字輪廓角落
    left, right, top, bottom = find_contour_corner(contours)
    
    #重塑文字大小
    img1 = Image.open(filename)
    img1 = img1.crop((left, top, right, bottom))
    img1 = img1.resize((130, 130))
    
    #創建符合模型格式的空白圖片
    target = Image.new('RGB', (256, 256), 'white')
    #貼上重塑大小後的文字
    target.paste(img1, (15, 50))
    #儲存圖片
    target.save(filename)


#將圖片資訊打包成obj檔
def pickle_examples(paths, val_path):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(val_path, 'wb') as fv:
        for p in paths:
            try:
                label = int(os.path.basename(p).split("_")[0])
                with open(p, 'rb') as f:
                    #print("img %s" % p, label)
                    img_bytes = f.read()
                    example = (label, img_bytes)
                    pickle.dump(example, fv)
            except ValueError:
                continue

#將要輸入給模型的圖片打包成obj檔
def packagetoobj(package_pic_dir, package_obj_savepath):
    dirpath=package_pic_dir
    val_path = os.path.join(package_obj_savepath, "val.obj")
    pickle_examples(sorted(glob.glob(os.path.join(dirpath, "*.jpg"))), val_path=val_path)


#圖片打包成obj檔案並使用模型產生圖片
def package_and_genera_pic(jpg_path):
    #空白圖片儲存路徑
    source_dir_path = "./source/"
    #文字圖片儲存資料夾
    pic_dir = "./test/"
    #obj檔案儲存資料夾
    obj_dir = './obj/'

    if not os.path.isdir(source_dir_path):
        os.mkdir(source_dir_path)

    if not os.path.isdir(pic_dir):
        os.mkdir(pic_dir)

    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)

    #圖片複製到指定資料夾
    for p in glob.glob(os.path.join(source_dir_path, "*.jpg")):
        os.remove(p)

    for p in glob.glob(os.path.join(pic_dir, "*.jpg")):
        os.remove(p)

    source_filename = source_dir_path+"0000.jpg"
    shutil.copyfile(jpg_path, source_filename)
    
    #調整文字尺寸
    resize_fontsize(source_filename)

    #創建符合模型格式的空白圖片
    shape = (256, 512, 3) # y, x, RGB
    img1 = Image.fromarray(np.full(shape, 255).astype(np.uint8))
    img1.save(source_dir_path+"0.jpg")
    img1 = Image.open(source_dir_path+"0.jpg")

    #如果寬或高大於256像素，將圖片resize
    img2 = Image.open(source_filename)
    img2_width = img2.size[0]
    img2_height = img2.size[1]
    if img2_width > 256 or img2_height > 256:
        if img2_width > img2_height:
            img2 = img2.resize((256, int(img2_height*(256/img2_width))))
        else:
            img2 = img2.resize((int(img2_width*(256/img2_height)), 256))

    #將resize後的圖片貼到空白圖片並儲存
    img1.paste(img2, (256, 0))
    img1.save(source_filename)

    #將文字圖片複製到指定資料夾
    label = 0
    for i in range(12):
        pic_name = str(label) + '_' + "0000.jpg"
        shutil.copyfile(source_filename ,pic_dir + pic_name)
        label += 1

    #圖片打包成obj檔案並使用模型產生圖片
    print("package pictures to obj.")
    packagetoobj(pic_dir, obj_dir)

def generate(input_dir, save_dir, input_path=None):
    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if input_path:
        package_and_genera_pic(input_path)
        print('generating images...')
        with torch.no_grad():
            main()

        #將生成圖片複製到指定路徑
        count = 0
        for i in range(12):
            source_dir = './infer/' + str(i) + '/'
            for p in glob.glob(os.path.join(source_dir, "*.png")):
                style_lable = i % 2
                save_file_name = save_dir + str(style_lable) + '_' + str(count) + '.png'
                img = Image.open(p)
                img = img.crop((0, 20, 170, 190))
                img.resize((256, 256))
                img.save(save_file_name)
                #shutil.copyfile(p, save_file_name)
                count += 1

    else:
        count = 0
        for p in glob.glob(os.path.join(input_dir, "*.jpg")):
            package_and_genera_pic(p)
            print('generating images...')
            with torch.no_grad():
                main()

            #將生成圖片複製到指定路徑
            for i in range(12):
                source_dir = './infer/' + str(i) + '/'
                for p in glob.glob(os.path.join(source_dir, "*.png")):
                    style_lable = i % 2
                    save_file_name = save_dir + str(style_lable) + '_' + str(count) + '.png'
                    img = Image.open(p)
                    img = img.crop((0, 20, 170, 190))
                    img.resize((256, 256))
                    img.save(save_file_name)
                    #shutil.copyfile(p, save_file_name)
                    count += 1

        for p in glob.glob(os.path.join(input_dir, "*.png")):
            package_and_genera_pic(p)
            print('generating images...')
            with torch.no_grad():
                main()

            #將生成圖片複製到指定路徑
            for i in range(12):
                source_dir = './infer/' + str(i) + '/'
                for p in glob.glob(os.path.join(source_dir, "*.png")):
                    style_lable = i % 2
                    save_file_name = save_dir + str(style_lable) + '_' + str(count) + '.png'
                    img = Image.open(p)
                    img = img.crop((0, 20, 170, 190))
                    img.resize((256, 256))
                    img.save(save_file_name)
                    #shutil.copyfile(p, save_file_name)
                    count += 1

#自選模型產生圖片儲存的路徑
generate_picture_choose_save_dir = ""
#圖片編號
pic_label = 0
#目前的模型生成圖片的路徑
show_generate_picture_path = args.save_dir + str(pic_label) +"_0.png"
#按鈕按下的顏色
button_activebackground = "#87CEFA"
#畫布長寬
canvas_width =300
canvas_height =300

#設定介面屬性
window = tk.Tk()
window.title('字體風格轉換')
align_mode = 'nswe'
pad = 5
div_size = 200

#定義每一個div的大小及欄位
div1 = tk.Frame(window,  width=div_size*2 , height=div_size/4 , bg='white')
div2 = tk.Frame(window,  width=div_size , height=div_size , bg='white')
div3 = tk.Frame(window,  width=div_size , height=div_size , bg='white')
div4 = tk.Frame(window,  width=div_size/2 , height=div_size*1.25 , bg='white')

div1.grid(column=0, row=0, padx=pad, pady=pad, columnspan=4, sticky=align_mode)
div2.grid(column=0, row=1, padx=pad, pady=pad, columnspan=2, rowspan=2, sticky=align_mode)
div3.grid(column=2, row=1, padx=pad, pady=pad, columnspan=2, rowspan=2, sticky=align_mode)
div4.grid(column=4, row=0, padx=pad, pady=pad, rowspan=3, sticky=align_mode)

#用於定義div的欄位數的function
def define_layout(obj, cols=1, rows=1):
    
    def method(trg, col, row):
        
        for c in range(cols):    
            trg.columnconfigure(c, weight=1)
        for r in range(rows):
            trg.rowconfigure(r, weight=1)

    if type(obj)==list:        
        [ method(trg, cols, rows) for trg in obj ]
    else:
        trg = obj
        method(trg, cols, rows)

define_layout(window, cols=5, rows=3)
define_layout([div1, div2, div3])

#變更按鈕狀態
def switchButtonState(button1):
    if (button1['state'] == tk.NORMAL):
        button1['state'] = tk.DISABLED
    else:
        button1['state'] = tk.NORMAL
        
#定義滑鼠控制繪圖的線條粗細
def paint(event):
    python_green = "#000000"
    x1, y1 = event.x, event.y
    if canvas_1.old_coords:
        x2, y2 = canvas_1.old_coords
        canvas_1.create_line(x1, y1, x2, y2, fill=python_green, capstyle=tk.ROUND, smooth=True, width=2)
    canvas_1.old_coords = x1, y1

def mouse_click(event):
    canvas_1.old_coords = (event.x, event.y)
    
def mouse_release(event):
    canvas_1.old_coords = None
    
#儲存檔案按鈕按鈕點擊事件
def button_save_canvas_picture_click_event():
    #將目前顯示在'圖片顯示區\n(文字轉換後)'的圖片儲存到使用者選擇的路徑
    jpg_dir_path = args.input_dir
    canvas_1.update()
    canvas_1.postscript(file="cavas_post.ps" ,colormode='color')
    im = Image.open("cavas_post.ps")
    jpg_dir_path = filedialog.asksaveasfilename(defaultextension = '.jpg')
    im.save(jpg_dir_path + '.jpg')
    
#清除畫布按鈕點擊事件
def button_clear_canvas_click_event():
    canvas_1.delete('all')

#儲存檔案按鈕按鈕點擊事件
def save_canvas_picture():
    #將目前顯示在'圖片顯示區\n(文字轉換後)'的圖片儲存到選擇的路徑
    jpg_dir_path = args.input_dir
    #確認資料夾是否存在，如不存在就創建
    if not os.path.isdir(jpg_dir_path):
        os.mkdir(jpg_dir_path)
    
    canvas_1.update()
    canvas_1.postscript(file="cavas_post.ps" ,colormode='color')
    im = Image.open("cavas_post.ps")
    jpg_dir_path = "./input/0.jpg"
    im.save(jpg_dir_path)

#選擇圖片按鈕點擊事件
def button_choose_char_path_click_envent():
    #重置lbl_1, lbl_2
    global lbl_1, lbl_2
    lbl_1.destroy()
    #lbl_2.destroy()
    
    from tkinter import Canvas
    from PIL import Image

    # 定义一个全局变量 jpg_dir_path
    global jpg_dir_path
    # 使用 filedialog.askopenfile() 函数打开一个文件对话框让用户选择一个文件
    dir_path = filedialog.askopenfile()
    # 获取选择的文件的路径
    jpg_dir_path = dir_path.name
    # 使用 PIL 的 Image.open 方法打开选择的图片文件
    img = Image.open(jpg_dir_path)

    # 獲取整張圖像的寬度和高度
    width, height = img.size

    # 設置 ROI 為整張圖像的邊界
    left = 0
    top = 0
    right = width
    bottom = height
    # 使用 crop 函数来获取 ROI
    roi = img.crop((left, top, right, bottom))

    # 将 ROI 转换为灰度图像
    gray = roi.convert('L')

    # 获取图片的宽度和高度
    width, height = gray.size

    # 使用 PIL 的 getdata 方法获取图片的所有像素的值，并将其转换为一个列表
    pixels = list(gray.getdata())

    # 计算图像在 Canvas 中的居中位置
    canvas_center_x = canvas_1.winfo_reqwidth() / 2
    canvas_center_y = canvas_1.winfo_reqheight() / 2

    # 计算图像在 Canvas 中居中显示的偏移量
    image_center_x = width / 2
    image_center_y = height / 2

    # 计算需要的偏移量
    offset_x = canvas_center_x - image_center_x
    offset_y = canvas_center_y - image_center_y

    # 在 Canvas 上绘制图像中的像素点并进行偏移
    for y in range(height):
        for x in range(width):
            # 获取当前像素的值
            pixel = pixels[y * width + x]
            # 在 Canvas 上绘制一个点，颜色由当前像素的灰度值决定，并进行偏移
            canvas_1.create_line(x + offset_x, y + offset_y, x + offset_x + 1, y + offset_y + 1, fill='#%02x%02x%02x' % (pixel, pixel, pixel))



    # 創建一個 Label 用於顯示文字
    lbl = tk.Label(window, text='圖片顯示區\n(文字轉換後)', bg='white', fg='#263238', font=('Arial', 12))
    # 將 Label 添加到 window 中
    lbl.grid(column=2, row=1)
    
    
#選擇資料夾按鈕點擊事件
def button_choose_char_dir_click_envent():
    #重置lbl_1, lbl_2
    global lbl_1, lbl_2
    lbl_1.destroy()
    #lbl_2.destroy()
    
    global jpg_dir_path
    dir_path = filedialog.askdirectory()
    jpg_dir_path = dir_path+"/"
    jpg_dir_filename = ""
    
    for p in glob.glob(os.path.join(jpg_dir_path, "*.jpg")):
        jpg_dir_filename = p
        break
    
    img = Image.open(jpg_dir_filename)
    imgTk =  ImageTk.PhotoImage(img)
    lbl_1 = tk.Label(window, image=imgTk)
    lbl_1.image = imgTk
    lbl_1.grid(column=0, row=1)
    """
    lbl_2 = tk.Label(window, text='圖片顯示區\n(文字轉換後)', bg='white', fg='#263238', font=('Arial', 12))
    lbl_2.grid(column=2, row=1)
    """
#第一次執行時清空指定資料夾
remove_state = 0

def create_dir():
    global remove_state
    #創建模型產生圖片的資料夾
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    else:
        if remove_state == 0:
            for p in glob.glob(os.path.join(args.save_dir, "*.png")):
                os.remove(p)
            remove_state = 1

#產生圖片按鈕點擊事件
def button_generate_pic_event():
    global jpg_dir_path, pic_label, show_generate_picture_path, button_open_dir_state, button_open_dir, lbl_2
    #變更按鈕言顏色
    button_generate_pic['bg'] = button_activebackground
    #確認資料夾是否存在，如不存在就創建，已經存在則清空該資料夾
    if not os.path.isdir(args.input_dir):
        os.mkdir(args.input_dir)
    
    #創建儲存生成圖片的資料夾
    create_dir()
    
    #儲存畫布成圖片
    save_canvas_picture()
    
    #生成圖片
    generate(args.input_dir, args.save_dir, args.input_path) 
    
    #顯示模型生成的圖片
    pic_label = 0
    show_generate_picture_path = args.save_dir + str(pic_label) +"_0.png"
    img = Image.open(show_generate_picture_path)
    imgTk =  ImageTk.PhotoImage(img)
    lbl_2.destroy()
    lbl_2 = tk.Label(window, image=imgTk)
    lbl_2.image = imgTk
    lbl_2.grid(column=2, row=1)
    
    #初次點擊按鈕變更最右邊一排按鈕的狀態
    if button_save_picture['state'] != tk.NORMAL:
        switchButtonState(button_open_dir)
        switchButtonState(button_save_picture)
        #如果顯示第一張圖片，則上一張按鈕設定為不能按
        button_choose_last_picture['state'] = tk.DISABLED
        button_choose_next_picture['state'] = tk.NORMAL
    else:
        button_choose_last_picture['state'] = tk.NORMAL
        button_choose_next_picture['state'] = tk.NORMAL
        
    #變更按鈕言顏色
    button_generate_pic['bg'] = 'SystemButtonFace'
    
#開啟資料夾按鈕點擊事件
def button_open_dir_click_event():
    subprocess.Popen('explorer %s'%(os.path.abspath(args.save_dir)))

#儲存檔案按鈕按鈕點擊事件
def button_save_picture_click_event():
    #將目前顯示在'圖片顯示區\n(文字轉換後)'的圖片儲存到使用者選擇的路徑
    global generate_picture_choose_save_dir, show_generate_picture_path
    generate_picture_choose_save_dir = filedialog.asksaveasfilename()
    shutil.copyfile(show_generate_picture_path,generate_picture_choose_save_dir)

#上一張按鈕點擊事件
def button_choose_last_picture_click_event():
    """
    模型生成圖片的檔名命名規則:
    範例: n1_n2.png
    n1: 風格編號
    n2: 模型生成轉換字體的編號
    """
    #如果下一張按鈕為不能按的狀態，則改變其狀態
    if button_choose_next_picture['state'] != tk.NORMAL:
        button_choose_next_picture['state'] = tk.NORMAL
    
    global show_generate_picture_path, lbl_2, pic_label
    filname_split_maxnum=11 #生成圖片最大編號
    filname_split = 0 #目前生成圖片編號
    filename = show_generate_picture_path #目前的模型生成圖片的路徑
    file_dir = os.path.dirname(filename) #目前的模型生成圖片路徑的目錄
    
    #取得檔名中每個片段的編號
    filname_split = int(os.path.basename(filename).split("_")[1].split(".")[0])
    
    #取得轉換文字的數量
    for p in glob.glob(os.path.join(file_dir, "*.png")):
        n = int(os.path.basename(p).split("_")[1].split(".")[0])
        if n > filname_split_maxnum:
            filname_split_maxnum = n
    
    #n1,n2編號變更
    if filname_split == 1:
        pic_label = 0
        filname_split = 10
    else:
        filname_split = filname_split - 2
    
    #設定圖片路徑
    filename = file_dir + '/' + str(pic_label) + '_' + str(filname_split) + '.png'
    show_generate_picture_path = filename
    
    #顯示目錄中目前圖片的上一張圖片
    img = Image.open(filename)
    imgTk =  ImageTk.PhotoImage(img)
    lbl_2.destroy()
    lbl_2 = tk.Label(window, image=imgTk)
    lbl_2.image = imgTk
    lbl_2.grid(column=2, row=1)
    
    #如果顯示第一張圖片，則按鈕設定為不能按
    if os.path.basename(filename)=="0_0.png":
        button_choose_last_picture['state'] = tk.DISABLED

#下一張按鈕點擊事件
def button_choose_next_picture_click_event():
    """
    模型生成圖片的檔名命名規則:
    範例: n1_n2.png
    n1: 風格編號
    n2: 模型生成轉換字體的編號
    """
    #如果上一張按鈕為不能按的狀態，則改變其狀態
    if button_choose_last_picture['state'] != tk.NORMAL:
        button_choose_last_picture['state'] = tk.NORMAL
    
    global show_generate_picture_path, lbl_2, pic_label
    filname_split_maxnum=11 #生成圖片最大編號
    filname_split = 0 #目前生成圖片編號
    filename = show_generate_picture_path #目前的模型生成圖片的路徑
    file_dir = os.path.dirname(filename) #目前的模型生成圖片路徑的目錄
    
    #取得檔名中每個片段的編號
    filname_split = int(os.path.basename(filename).split("_")[1].split(".")[0])
    
    #取得轉換文字的數量
    for p in glob.glob(os.path.join(file_dir, "*.png")):
        n = int(os.path.basename(p).split("_")[1].split(".")[0])
        if n > filname_split_maxnum:
            filname_split_maxnum = n
    
    #n1,n2編號變更
    if filname_split == filname_split_maxnum - 1:
        pic_label = 1
        filname_split = 1
    else:
        filname_split = filname_split + 2
    
    #設定圖片路徑
    filename = file_dir + '/' + str(pic_label) + '_' + str(filname_split) + '.png'
    show_generate_picture_path = filename
    
    #顯示目錄中目前圖片的上一張圖片
    img = Image.open(filename)
    imgTk =  ImageTk.PhotoImage(img)
    lbl_2.destroy()
    lbl_2 = tk.Label(window, image=imgTk)
    lbl_2.image = imgTk
    lbl_2.grid(column=2, row=1)
    
    #如果顯示最後一張圖片，則按鈕設定為不能按
    if filname_split == filname_split_maxnum:
        button_choose_next_picture['state'] = tk.DISABLED

#設定介面內容

#選擇圖片按鈕    
button_choose_char_path = tk.Button(window, text="選擇圖片", fg='black', font=('Arial', 12), 
                                    command=button_choose_char_path_click_envent, activebackground=button_activebackground)
button_choose_char_path.grid(column=5, row=0)
"""
#選擇資料夾按鈕
button_choose_char_dir = tk.Button(window, text="選擇資料夾", fg='black', font=('Arial', 12), 
                                   command=button_choose_char_dir_click_envent, activebackground=button_activebackground)
button_choose_char_dir.grid(column=1, row=0)
"""
#產生圖片按鈕
button_generate_pic = tk.Button(window, text="產生圖片", fg='black', font=('Arial', 12), 
                 command=button_generate_pic_event, activebackground=button_activebackground)
button_generate_pic.grid(column=2, row=0)

#開啟儲存模型產生圖片的資料夾
button_open_dir = tk.Button(window, text="開啟資料夾", fg='black', font=('Arial', 12), 
                            command=button_open_dir_click_event, activebackground=button_activebackground)
button_open_dir.grid(column=3, row=0)
switchButtonState(button_open_dir)

#圖片顯示區
lbl_1 = tk.Label(window, text='圖片顯示區', bg='white', fg='#263238', font=('Arial', 12))
lbl_1.grid(column=0, row=1)

#圖片(文字轉換後)顯示區
lbl_2 = tk.Label(window, text='圖片顯示區\n(文字轉換後)', bg='white', fg='#263238', font=('Arial', 12))
lbl_2.grid(column=2, row=1)

#定義畫布

# 創建一個 Canvas 對象，設定其寬度、高度和背景顏色，並將其添加到 window 中
canvas_1 = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas_1.grid(column=0, row=1)

# 設定一個屬性來存儲舊的座標
canvas_1.old_coords = None
# 綁定滑鼠左鍵拖動事件到 paint 函數
canvas_1.bind("<B1-Motion>", paint)
# 綁定滑鼠左鍵點擊事件到 mouse_click 函數
canvas_1.bind("<Button-1>", mouse_click)
# 綁定滑鼠左鍵釋放事件到 mouse_release 函數
canvas_1.bind("<ButtonRelease-1>", mouse_release)

#畫布訊息
message = tk.Label(window, text="滑鼠點擊繪圖", bg='white')
message.grid(column=0, row=1,sticky='n')
 
#儲存畫布按鈕
button_save_canvas_picture = tk.Button(window, text="儲存畫布", fg='black', font=('Arial', 12), 
                                    command=button_save_canvas_picture_click_event, activebackground=button_activebackground)
button_save_canvas_picture.grid(column=0, row=0)

#清除畫布按鈕
button_clear_canvas = tk.Button(window, text="清除畫布", fg='black', font=('Arial', 12), 
                                    command=button_clear_canvas_click_event, activebackground=button_activebackground)
button_clear_canvas.grid(column=1, row=0)

#儲存檔案按鈕
button_save_picture = tk.Button(window, text="儲存圖片", fg='black', font=('Arial', 12), 
                                    command=button_save_picture_click_event, activebackground=button_activebackground)
button_save_picture.grid(column=4, row=0)
switchButtonState(button_save_picture)

#上一張按鈕
button_choose_last_picture = tk.Button(window, text="上一張", fg='black', font=('Arial', 12), 
                                    command=button_choose_last_picture_click_event, activebackground=button_activebackground)
button_choose_last_picture.grid(column=4, row=1)
switchButtonState(button_choose_last_picture)

#下一張按鈕
button_choose_next_picture = tk.Button(window, text="下一張", fg='black', font=('Arial', 12), 
                                    command=button_choose_next_picture_click_event, activebackground=button_activebackground)
button_choose_next_picture.grid(column=4, row=2)
switchButtonState(button_choose_next_picture)

window.mainloop()
#generate(args.input_dir, args.save_dir, args.input_path)