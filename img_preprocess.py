import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw


def rgb2gray(img_path: str, K=8, height=128):
    img = cv2.imread(img_path)
    source_size = img.shape
    img = cv2.resize(img, (height, int(height * source_size[1] / source_size[0])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Z = img.reshape((-1, 1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 给center排序
    sorted_center = np.argsort(center, axis=0)
    label = label.reshape((img.shape))
    img_dict = {
        # sorted_center[0][0]: "░░",
        # sorted_center[1][0]: "▒▒",
        # sorted_center[2][0]: "▓▓",
        # sorted_center[3][0]: "██",

        # sorted_center[0][0]: "囗",
        # sorted_center[1][0]: "园",
        # sorted_center[2][0]: "圈",
        # sorted_center[3][0]: "圚",

        sorted_center[0][0]: "  ",
        sorted_center[1][0]: "囗",
        sorted_center[2][0]: "四",
        sorted_center[3][0]: "园",
        sorted_center[4][0]: "囿",
        sorted_center[5][0]: "圈",
        sorted_center[6][0]: "圓",
        sorted_center[7][0]: "圚",

        # sorted_center[0][0]: "  ",
        # sorted_center[1][0]: "囗",
        # sorted_center[2][0]: "四",
        # sorted_center[3][0]: "因",
        # sorted_center[4][0]: "园",
        # sorted_center[5][0]: "图",
        # sorted_center[6][0]: "囿",
        # sorted_center[7][0]: "圆",
        # sorted_center[8][0]: "圈",
        # sorted_center[9][0]: "圍",
        # sorted_center[10][0]: "圓",
        # sorted_center[11][0]: "團",
        # sorted_center[12][0]: "圚",
        # sorted_center[13][0]: "圜",
        # sorted_center[14][0]: "圝",
        # sorted_center[15][0]: "圞",
    }

    res = np.vectorize(img_dict.get)(label)
    return res


def text2png(res, font_path: str, font_size: int = 6, font_color="#555", bg_color="#FFF"):
    font = ImageFont.truetype(font_path, font_size)
    ascent, descent = font.getmetrics()
    s = ""
    for l in res[0]:
        s = s + str(l)

    _, _, w, h = font.getmask(s).getbbox()
    img = Image.new("RGB", (w, h * len(res)), bg_color)
    draw = ImageDraw.Draw(img)
    y = 0
    for raw_line in res:
        line = ""
        for l in raw_line:
            line = line + str(l)

        draw.text((0, y), line, font=font, fill=font_color)
        y += h
        print(line)

    img.save("res.png")

    # draw = ImageDraw.Draw(img)

    # lines = text.split('\n')
    #
    # for line in lines:
    #     print(line)


# show time
if __name__ == "__main__":
    text = rgb2gray("ziggy.jpg")
    text2png(text, "SarasaMonoSC-Regular.ttf", font_size=14)
