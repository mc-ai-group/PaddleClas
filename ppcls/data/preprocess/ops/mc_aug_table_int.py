import cv2
import numpy as np
import random


class McAugTableInt(object):
    def __init__(
        self,
        giveup=0.3,
        zoom=0.1,
        salt=0.2,
    ):
        self.giveup = giveup
        self.zoom = zoom
        self.salt = salt

    def __call__(self, img):
        if random.random() < self.giveup:
            return img
        ori_size = img.shape[0]
        img_out = img.copy()

        # 随机放大
        border = random.randint(1, round(ori_size * self.zoom))
        img_out = cv2.resize(
            img_out,
            (ori_size + border * 2, ori_size + border * 2),
            interpolation=cv2.INTER_AREA & cv2.INTER_MAX,
        )
        img_out = img_out[border : border + ori_size, border : border + ori_size]

        # 加椒盐噪点
        s = range(0, ori_size)
        k = round(len(s) * self.salt)
        xs = random.sample(s, k)
        ys = random.sample(s, k)
        cs = random.sample(range(0, 255), k)
        img_out[ys, xs] = [[c]*3 for c in cs]

        # 加文字干扰
        if random.randint(0, 1):
            text = str(random.randint(1e3, 1e4))
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3 + random.random() * 0.4
            thickness = 2
            (w, h), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            x = random.randint(0-w, ori_size)
            y = random.randint(0, ori_size+h)
            cv2.putText(
                img_out, text, (x, y), font_face, font_scale, [random.randint(0, 200)]*3, thickness
            )

        return img_out


if __name__ == "__main__":
    import os

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    file = os.path.abspath(
        os.path.join(
            __dir__,
            "..",
            "..",
            "..",
            "..",
            "..",
            "ocr-train-data",
            "table_int_clas",
            "train",
            "5",
            "002-05-405_349_4_3-50.png",
        )
    )
    out_dir = "D:\\debug\\mc_aug_table_int"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    mc_aug_table_int = McAugTableInt()
    img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
    for i in range(10):
        out_file = os.path.join(out_dir, f"{i}.png")
        cv2.imencode(".png", mc_aug_table_int(img))[1].tofile(out_file)
