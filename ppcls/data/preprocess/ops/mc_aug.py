import cv2
import numpy as np
import random


class McAug(object):
    def __init__(
        self,
        giveup=0.3,
        angle=15,
        resize=0.8,
        salt=0.2,
        distort_lines=20,
        distort_circles=20,
    ):
        self.giveup = giveup
        self.angle = angle
        self.resize = resize
        self.salt = salt
        self.distort_lines = distort_lines
        self.distort_circles = distort_circles

    def __call__(self, img):
        if random.random() < self.giveup:
            return img
        ori_w = img.shape[1]
        ori_h = img.shape[0]
        # 随机旋转
        angle = random.random() * 2 * self.angle - self.angle
        M = cv2.getRotationMatrix2D((ori_w // 2, ori_h // 2), angle, 1)
        img = cv2.warpAffine(
            img, M, (ori_w, ori_h), borderValue=0, flags=cv2.INTER_CUBIC
        )

        # 随机缩放
        if random.random() >= 0.5:
            ratio_w = self.resize + (1 - self.resize) * random.random()
            ratio_h = 1
        else:
            ratio_w = 1
            ratio_h = self.resize + (1 - self.resize) * random.random()
        h = round(ori_h * ratio_h)
        w = round(ori_w * ratio_w)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA & cv2.INTER_MAX)
        y_start = (ori_h - img.shape[0]) // 2
        x_start = (ori_w - img.shape[1]) // 2
        img[img > 20] = 255
        img_out = np.zeros((ori_w, ori_h, 3), dtype=np.uint8)
        img_out[
            y_start : y_start + img.shape[0], x_start : x_start + img.shape[1]
        ] = img

        # 随机平移
        dist = 10
        M = np.float32(
            [
                [1, 0, dist - round(2 * dist * random.random())],
                [0, 1, dist - round(2 * dist * random.random())],
            ]
        )
        img_out = cv2.warpAffine(
            img_out, M, (ori_w, ori_h), borderValue=0, flags=cv2.INTER_CUBIC
        )

        # 加椒盐噪点
        xs = range(0, ori_w)
        ys = range(0, ori_h)
        xs = random.sample(xs, round(len(xs) * self.salt))
        ys = random.sample(ys, round(len(ys) * self.salt))
        img_out[ys, xs] = 255

        # 加短线干扰
        for i in range(self.distort_lines):
            x1 = round(ori_w * random.random())
            y1 = round(ori_h * random.random())
            x2 = np.clip(
                x1 + round(ori_w * (0.05 * random.random() - 0.025)), 0, ori_w - 1
            )
            y2 = np.clip(
                y1 + round(ori_h * (0.05 * random.random() - 0.025)), 0, ori_h - 1
            )
            cv2.line(img_out, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)

        # 加小圆干扰
        for i in range(self.distort_circles):
            x = round(ori_w * random.random())
            y = round(ori_h * random.random())
            r = round(min(ori_w, ori_h) * 0.02 * random.random())
            cv2.circle(img_out, (x, y), r, (255, 255, 255), thickness=1)

        # 加文字干扰
        text_count = random.randint(0, 1)
        for i in range(text_count):
            x = round(ori_w * random.random())
            y = round(ori_h * random.random())
            text = str(random.randint(1e3, 1e5))
            font_size = 0.3 + random.random() * 0.4
            cv2.putText(
                img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255)
            )

        return img_out
