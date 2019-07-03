import cv2
import numpy as np
from settings import Args

def narrow_rect(contours):
    """
    :param contours:
    :return: dict dict
    """

    rect_S_dict = {}
    rect_dict = {}

    for i in range(0, len(contours)):
        if len(contours[i]) > 0:

            # remove small objects
            if cv2.contourArea(contours[i]) < 1000:
                continue

            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)

            rect_dict[i] = [x, y, w, h]
            S = w * h
            rect_S_dict[i] = S

    return rect_dict, rect_S_dict

def concat_rect(rect_dict, rect_S_dict, num_blackboard):
    """
    :param rect_dict: dict
    :param rect_S_dict: dict
    :param num_blackboard: int
    :return: int*4
    """

    counter = 0
    minx = miny = 1000000
    max_x = max_y = 0
    for k, v in sorted(rect_S_dict.items(), key=lambda x: -x[1]):
        x = rect_dict[k][0]
        y = rect_dict[k][1]
        w = rect_dict[k][2]
        h = rect_dict[k][3]

        if minx >= x:
            minx = x
        if miny >= y:
            miny = y
        if max_x <= x + w:
            max_x = x + w
        if max_y <= y + h:
            max_y = y + h

        counter += 1
        if counter >= num_blackboard:
            return minx, miny, max_x, max_y



def detect_brackboard(img, num_blackboard, thres=50):
    """
    :param img: array
    :return: int*4
    """

    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 指定した色に基づいたマスク画像の生成
    img_mask = cv2.inRange(img_HSV, lower, upper)

    # フレーム画像とマスク画像の共通の領域を抽出する。
    img_color = cv2.bitwise_and(img, img, mask=img_mask)

    # gray変換
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 閾値
    _thre, img_th = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)

    # find tulips
    labels, contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # rectを絞り、面積を計算
    rect_dict, rect_S_dict = narrow_rect(contours)

    # rectを結合し黒板を抽出
    minx, miny, max_x, max_y = concat_rect(rect_dict, rect_S_dict, num_blackboard)

    return minx, miny, max_x, max_y


if __name__ == '__main__':

    # 取得する色の範囲を指定する
    lower = np.array(Args['LOW_RANGE'])
    upper = np.array(Args['HIGH_RANGE'])

    # キャプションの作成
    cap = cv2.VideoCapture(Args['INPUT_PATH'])
    # キャプション情報の取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    Width = int(cap.get(3))
    Height = int(cap.get(4))

    # VideoWriter を作成する。
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # 黒板の座標を取得
    minx, miny, max_x, max_y = detect_brackboard(cap.read()[1], Args['NUM_BLACKBOARD'], Args['THRES'])

    # 動画のサイズを取得
    size = (int((max_x - minx)/Args['X_SIZE_RATIO']), int((max_y - miny)/Args['Y_SIZE_RATIO']))

    # writerを作成
    writer = cv2.VideoWriter(Args['OUTPUT_PATH'], fourcc, fps, size)

    # 初期化
    cap = cv2.VideoCapture(Args['INPUT_PATH'])

    while (cap.isOpened()):

        ret, img = cap.read()

        if ret == False:
            break

        # 黒板を切り出し
        img_clip = img[miny:max_y, minx:max_x, :]
        img_clip = cv2.resize(img_clip, size)

        writer.write(img_clip)  # フレームを書き込む。
        # cv2.imshow("img_clipped", img_clip)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    writer.release()

