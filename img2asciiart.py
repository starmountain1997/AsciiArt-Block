import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw


def img2txt(img_path: str, K=4, height=None):
    img = cv2.imread(img_path)
    source_size = img.shape
    if height is not None:
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
        sorted_center[0][0]: "░░",
        sorted_center[1][0]: "▒▒",
        sorted_center[2][0]: "▓▓",
        sorted_center[3][0]: "██",

        # sorted_center[0][0]: "  ",
        # sorted_center[1][0]: "囗",
        # sorted_center[2][0]: "目",
        # sorted_center[3][0]: "眼",

        # sorted_center[0][0]: "  ",
        # sorted_center[1][0]: "天",
        # sorted_center[2][0]: "京",
        # sorted_center[3][0]: "祺",
        # sorted_center[4][0]: "憶",
        # sorted_center[5][0]: "礦",
        # sorted_center[6][0]: "囕",
        # sorted_center[7][0]: "𰻞",

        # sorted_center[0][0]: "  ",
        # sorted_center[1][0]: "二",
        # sorted_center[2][0]: "天",
        # sorted_center[3][0]: "丟",
        # sorted_center[4][0]: "京",
        # sorted_center[5][0]: "祟",
        # sorted_center[6][0]: "祺",
        # sorted_center[7][0]: "慈",
        # sorted_center[8][0]: "憶",
        # sorted_center[9][0]: "嚙",
        # sorted_center[10][0]: "礦",
        # sorted_center[11][0]: "鑑",
        # sorted_center[12][0]: "囕",
        # sorted_center[13][0]: "虌",
        # sorted_center[14][0]: "鸛",
        # sorted_center[15][0]: "𰻞",
    }

    res = np.vectorize(img_dict.get)(label)
    np.savetxt("res.txt", res, delimiter="", fmt="%s")
    return res


def save_txt_as_img(res, font_path: str, font_size: int = 6, font_color="#DC143C"):
    """

    :param res: input np.array
    :param font_path:
    :param font_size:
    :param font_color:
    :return:
    """
    font = ImageFont.truetype(font_path, font_size)
    ascent, descent = font.getmetrics()
    s = ""
    for l in res[0]:
        s = s + str(l)

    _, _, w, h = font.getmask(s).getbbox()
    img = Image.new("RGB", (w, h * len(res)), "#FFF")
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


# show time
if __name__ == "__main__":
    text = img2txt("ziggy.jpg")
    save_txt_as_img(text, "SarasaMonoSC-Regular.ttf", font_size=16)
