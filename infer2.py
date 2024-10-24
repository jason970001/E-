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
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from model.model import chk_mkdir


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
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--batch_size', type=int, default=1, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=34000, help='resume from previous training')
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
    img = Image.new("L", (canvas_size, canvas_size), 255)  # 使用 "L" 模式創建灰度圖像
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, 0, font=font)  # 使用黑色 (0) 繪製文字
    return img

def main():
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    chk_mkdir(infer_dir)

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
            writer_dict_inv = {v: k for k, v in writer_dict.items()}
            for label_idx in range(29):
                model.set_input(torch.ones_like(batch[0]) * label_idx, batch[2], batch[1])
                model.forward()
                tensor_to_plot = torch.cat([model.fake_B, model.real_B], 2)
                save_image(tensor_to_plot, os.path.join(infer_dir, "infer_{}".format(writer_dict_inv[label_idx]) + "_construct.png"))
        else:
            model.sample(batch, infer_dir)
            global_steps += 1

    t_finish = time.time()

    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))

def create_dir(val_obj_path):
    global remove_state
    if not os.path.isdir(val_obj_path):
        os.mkdir(val_obj_path)

def pickle_examples(paths, val_path):
    with open(val_path, 'wb') as fv:
        for p in paths:
            try:
                label = int(os.path.basename(p).split("_")[0])
                with open(p, 'rb') as f:
                    img_bytes = f.read()
                    example = (label, img_bytes)
                    pickle.dump(example, fv)
            except ValueError:
                continue

def packagetoobj(package_pic_dir, package_obj_savepath):
    dirpath = package_pic_dir
    val_path = os.path.join(package_obj_savepath, "val.obj")
    pickle_examples(sorted(glob.glob(os.path.join(dirpath, "*.png"))), val_path=val_path)

def package_and_genera_pic(png_path):
    source_dir_path = "./source/"
    pic_dir = "./test/"
    obj_dir = './obj/'

    if not os.path.isdir(source_dir_path):
        os.mkdir(source_dir_path)

    if not os.path.isdir(pic_dir):
        os.mkdir(pic_dir)

    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)

    for p in glob.glob(os.path.join(source_dir_path, "*.png")):
        os.remove(p)

    for p in glob.glob(os.path.join(pic_dir, "*.png")):
        os.remove(p)

    source_filename = source_dir_path + "0000.png"
    shutil.copyfile(png_path, source_filename)

    shape = (256, 512, 3)  # y, x, RGB
    img1 = Image.fromarray(np.full(shape, 255).astype(np.uint8))
    img1.save(source_dir_path + "0.png")
    img1 = Image.open(source_dir_path + "0.png")

    # 如果寬或高大於 256 像素，將圖片 resize
    img2 = Image.open(source_filename)
    img2_width = img2.size[0]
    img2_height = img2.size[1]
    if img2_width > 256 or img2_height > 256:
        if img2_width > img2_height:
            img2 = img2.resize((256, int(img2_height * (256 / img2_width))))
        else:
            img2 = img2.resize((int(img2_width * (256 / img2_height)), 256))

    # 將 resize 後的圖片貼到空白圖片並儲存
    img1.paste(img2, (256, 0))
    
    img1.save(source_filename)

    # 將文字圖片複製到指定資料夾
    label = 0
    for i in range(12):
        pic_name = str(label) + '_' + "0000.png"
        shutil.copyfile(source_filename, pic_dir + pic_name)
        label += 1

    # 圖片打包成 obj 檔案並使用模型產生圖片
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

        # 將生成圖片複製到指定路徑
        count = 0
        for i in range(12):
            source_dir = './infer/' + str(i) + '/'
            for p in glob.glob(os.path.join(source_dir, "*.png")):
                save_file_name = os.path.join(save_dir, f'{count}.png')
                img = Image.open(p)
                #img = img.convert("L")  # 將圖像轉換為灰度圖像
                r, g, b = img.split()  # 提取紅色通道
                img = r
                img.save(save_file_name)
                count += 1

    else:
        count = 0
        for p in glob.glob(os.path.join(input_dir, "*.png")):
            package_and_genera_pic(p)
            print('generating images...')
            with torch.no_grad():
                main()

            # 將生成圖片複製到指定路徑
            for i in range(12):
                source_dir = './infer/' + str(i) + '/'
                for p in glob.glob(os.path.join(source_dir, "*.png")):
                    save_file_name = os.path.join(save_dir, f'{count}.png')
                    img = Image.open(p)
                    img = img.crop((0, 0, 256, 256))
                    #img = img.convert("L")  # 將圖像轉換為灰度圖像
                    r, g, b = img.split()  # 提取紅色通道
                    img = r
                    img.save(save_file_name)
                    count += 1


generate(args.input_dir, args.save_dir, args.input_path)