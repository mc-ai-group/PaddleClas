import cv2
import numpy as np
import random


class McAugRailwayBridge(object):
    def __init__(
        self,
        giveup=0.3,
        zoom=0.02,
        salt=0.5,
    ):
        self.giveup = giveup
        self.zoom = zoom
        self.salt = salt

    def __call__(self, img):
        # if random.random() < self.giveup:
        #     return img
        ori_x_size = img.shape[0]
        ori_y_size = img.shape[1]
        ori_max_size = max(ori_x_size, ori_y_size)
        ori_min_size = min(ori_x_size, ori_y_size)
        img_out = img.copy()

        # 随机放大
        border = random.randint(1, round(ori_max_size * self.zoom))
        img_out = cv2.resize(
            img_out,
            (ori_y_size + border * 2, ori_x_size + border * 2),
            interpolation=cv2.INTER_AREA & cv2.INTER_MAX,
        )
        img_out = img_out[border : border + ori_x_size, border : border + ori_y_size]

        # 加椒盐噪点
        s = range(0, ori_min_size)
        k = round(len(s) * self.salt)
        xs = random.sample(s, k)
        ys = random.sample(s, k)
        cs = [random.randint(0, 255) for _ in range(k)]
        img_out[ys, xs] = [[c]*3 for c in cs]

        # 加文字干扰
        if random.randint(0, 1):
            text = str(random.randint(1e3, 1e4))
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3 + random.random() * 0.4
            thickness = 2
            (w, h), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            x = random.randint(0-w, ori_min_size)
            y = random.randint(0, ori_min_size+h)
            cv2.putText(
                img_out, text, (x, y), font_face, font_scale, [random.randint(0, 200)]*3, thickness
            )

        return img_out

if __name__ == "__main__":
    import os

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    folder = r'C:\Users\zxq\Desktop\新建文件夹 (2)\bearing_platform'
    filelist = os.listdir(folder)
    out_dir = r'C:\Users\zxq\Desktop\生成图片'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    mc_aug_table_int = McAugRailwayBridge()

    gen_count = 35

    for i in range(gen_count):
        random_file = random.choice(filelist)
        img = cv2.imdecode(np.fromfile(os.path.join(folder, random_file), dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imencode(".png", mc_aug_table_int(img))[1].tofile(os.path.join(out_dir, f"{random_file[:-4]}_{i}.png"))
    