import  cv2
import  numpy as np

def compare(sample=None,testImg=None):
    if(sample is None):
        sample = cv2.imread("./data/monoEraser2.jpg")
        testImg = cv2.imread("./data/test2.jpg")
    # A-KAZE検出器の生成
    akaze = cv2.AKAZE_create()

    # 特徴量の検出と特徴量ベクトルの計算
    kp1, des1 = akaze.detectAndCompute(sample, None)
    kp2, des2 = akaze.detectAndCompute(testImg, None)

    # Brute-Force Matcher生成
    bf = cv2.BFMatcher()

    # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
    matches = bf.knnMatch(des1, des2, k=2)

    # データを間引きする
    ratio = 0.5
    good = []
    for m, n in matches:
        #if m.distance < ratio * n.distance:
        good.append([m])


    print(len(good))

    # 対応する特徴点同士を描画
    img3 = cv2.drawMatchesKnn(sample, kp1, testImg, kp2, good, None, flags=2)

    # 画像表示
    cv2.imshow('img', img3)
    #cv2.imwrite("./data/output5.png", img3)
    # キー押下で終了
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if(len(good)>300):
        return  True
    else:
        return  False


def rotateImg(originalImg,angle):
    height, width, channels = originalImg.shape
    # getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D((int(width/2), int(height/2)), angle, 1)
    # アフィン変換
    image2 = cv2.warpAffine(originalImg, trans, (width, height))


if __name__ == '__main__':
    compare()