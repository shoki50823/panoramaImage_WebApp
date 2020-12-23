import cv2
import numpy as np
import copy


class Image:
    def __init__(self,name,img):
        def calculate(self):
            # 特徴点の検出手法
            detector = cv2.AKAZE_create()

            # 特徴点の検出手法
            # キーポイントと特徴量を出力
            keypoints, descriptors = detector.detectAndCompute(self.image,None)
            return keypoints, descriptors

        self.name = name
        self.image = img
        self.kp,self.des = calculate(self)

    def show(self):
        cv2.imshow(self.name, self.image)
        cv2.waitKey(0)

    def resize_mat(self, div):
        height, width = self.image.shape[0:2]
        d = [0, 0, width, height]
        if div[0][0] < 0:
            d[0] = div[0][0]
        if div[0][1] > width:
            d[2] = div[0][1]
        if div[1][0] < 0:
            d[1] = div[1][0]
        if div[1][1] > height:
            d[3] = div[1][1]
        T = np.array([[1.0, 0.0, -d[0]], [0.0, 1.0, -d[1]], [0.0, 0.0, 1.0]])
        # T = np.array([[1.0, 0.0, -d[1]], [0.0, 1.0, -d[0]], [0.0, 0.0, 1.0]])
        # 画像を平行移動＆リサイズ(画像の拡大縮小はない．合成画像が入るための余白を追加)
        # d[0]d[1]は0またはマイナスの座標であるため画像サイズにはマイナスして足す
        # self.image = cv2.warpPerspective(self.image, T, (int(-d[0] + d[2]), int(-d[1] + d[3])))
        self.image = warpPerspective(self.image, T, (int(-d[0] + d[2]), int(-d[1] + d[3])))
        # cv2.imwrite(".result/resize_img_.png",self.image)
        return d

def resize_image(img):
    img = cv2.resize(img, dsize=(int(600),int(800)))
    for i in img:
        for j in i:
            # 真っ黒な画素は全色+1ずつ
            if not j.all():
                j[0] += 1
                j[1] += 1
                j[2] += 1
    return img

def calc_dst4points(H, size):
    x = []
    y = []
    x.append(((H[0][0]*0 + H[0][1]*0 + H[0][2])/(H[2][0]*0 + H[2][1]*0 + H[2][2])))
    y.append(((H[1][0]*0 + H[1][1]*0 + H[1][2])/(H[2][0]*0 + H[2][1]*0 + H[2][2])))
    x.append(((H[0][0]*0 + H[0][1]*size[0] + H[0][2])/(H[2][0]*0 + H[2][1]*size[0] + H[2][2])))
    y.append(((H[1][0]*0 + H[1][1]*size[0] + H[1][2])/(H[2][0]*0 + H[2][1]*size[0] + H[2][2])))
    x.append(((H[0][0]*size[1] + H[0][1]*0 + H[0][2])/(H[2][0]*size[1] + H[2][1]*0 + H[2][2])))
    y.append(((H[1][0]*size[1] + H[1][1]*0 + H[1][2])/(H[2][0]*size[1] + H[2][1]*0 + H[2][2])))
    x.append(((H[0][0]*size[1] + H[0][1]*size[0] + H[0][2])/(H[2][0]*size[1] + H[2][1]*size[0] + H[2][2])))
    y.append(((H[1][0]*size[1] + H[1][1]*size[0] + H[1][2])/(H[2][0]*size[1] + H[2][1]*size[0] + H[2][2])))

    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)
    div = ((min_x, max_x),(min_y, max_y))
    return div

def write_blending(target, source, SrcMask):
    mask = cv2.cvtColor(SrcMask,cv2.COLOR_GRAY2RGB)
    target[(mask != [0,0,0])] = source[(mask != [0,0,0])]
    return target


def make_mask(target, src):
    # パノラマ画像のサイズで配列を初期化
    CommonMaskRGB = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
    SrcMaskRGB = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
    TargetMaskRGB = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)

    # 射影変換した画像と合成したい画像の積を出して，画像が重なる部分のマスクを出している
    CommonMaskRGB[(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY) != 0) * (cv2.cvtColor(src,cv2.COLOR_RGB2GRAY) != 0)] = 255
    
    SrcMaskRGB[(cv2.cvtColor(src,cv2.COLOR_RGB2GRAY) != 0) * (CommonMaskRGB == 0)] = 255
    TargetMaskRGB[(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY) != 0)] = 255

    # 収縮
    # CommonMaskRGBにおいて5*5の範囲でが全て1なら1を返す．この処理を3回行う．
    CommonMask = cv2.erode(CommonMaskRGB,np.ones((5,5),np.uint8),iterations = 3)
    SrcMask = cv2.dilate(SrcMaskRGB,np.ones((5,5),np.uint8),iterations = 1)
    TargetMask = cv2.dilate(TargetMaskRGB,np.ones((3,3),np.uint8),iterations = 1)
    return CommonMask, SrcMask, TargetMask

def arrange_rgb(mat, TargetMask):
    mat[TargetMask==0] = [0,0,0]
    gray = cv2.cvtColor(mat,cv2.COLOR_RGB2GRAY)
    mat[(TargetMask != 0) * (gray == 0)] = 1
    return mat

def get_center(mask):
    min_x = 10000
    max_x = -1
    min_y = 10000
    max_y = -1
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if(mask[y][x]):
                if(x<min_x):
                    min_x = x
                if(y<min_y):
                    min_y = y
                if(x>max_x):
                    max_x = x
                if(y>max_y):
                    max_y = y
    return (max_y+min_y)/2, (max_x+min_x)/2

# def warpPerspective(src,H,shape):
#     panorama = np.zeros((shape[1], shape[0],3), dtype=np.uint8)
#     # panorama = np.zeros((shape[1], shape[0],3), dtype=np.float32)
#     # panorama = np.zeros((1000, 1000,3), dtype=np.float32)
    

#     for i in range(src.shape[0]):
#         for j in range(src.shape[1]):
#             # vec = np.array([float(i),float(j),1.0])
#             # vec_ = np.dot(H,vec)
#             vec_ = np.array([0,0])

#             vec_[0] = ((H[0][0]*i + H[0][1]*j + H[0][2])/(H[2][0]*i + H[2][1]*j + H[2][2]))
#             vec_[1] = ((H[1][0]*i + H[1][1]*j + H[1][2])/(H[2][0]*i + H[2][1]*j + H[2][2]))

#             if int(round(vec_[0])) >= shape[1]:
#                 vec_[0] = shape[1]-1

#             if int(round(vec_[1])) >= shape[0]:
#                 vec_[1] = shape[0]-1

#             panorama[int(round(vec_[0]))][int(round(vec_[1]))] = src[i][j]
#             # print("panorama[" + str(int(vec_[0])) + "][" + str(int(vec_[1])) + "]=" + str(panorama[int(vec_[0])][int(vec_[1])]))
#             # print("src" + str(i) + str(j) +" = " + str(src[i][j]))

#     return panorama

# def warpPerspective(src,H,shape):
#     H_inv = np.linalg.inv(H)
#     panorama = np.zeros((shape[1], shape[0],3), dtype=np.uint8)

#     vec_ = np.array([0,0])

#     for i in range(panorama.shape[0]):
#         for j in range(panorama.shape[1]):
#             # vec = np.array([j,i,1])
#             # vec_ = np.dot(H_inv,vec).astype('int8')

#             vec_[0] = ((H_inv[1][0]*j + H_inv[1][1]*i + H_inv[1][2])/(H_inv[2][0]*j + H_inv[2][1]*i + H_inv[2][2]))
#             vec_[1] = ((H_inv[0][0]*j + H_inv[0][1]*i + H_inv[0][2])/(H_inv[2][0]*j + H_inv[2][1]*i + H_inv[2][2]))

#             # if int(round(vec_[0])) < src.shape[0] and int(round(vec_[1])) < src.shape[1]:
#             #     panorama[i][j] = src[int(round(vec_[0]))][int(round(vec_[1]))]

#             if vec_[0] < src.shape[0] and vec_[1] < src.shape[1] and vec_[0] >= 0 and vec_[1] >= 0:
#                 panorama[i][j] = src[vec_[0]][vec_[1]]
#                 # print("src[" + str(vec_[0]) + "][" + str(vec_[1]) +"] = " + str(src[vec_[0]][vec_[1]]))

#             else:
#                 panorama[i][j] = 0

#             # try:
#             #     src[vec_[0]][vec_[1]]
#             # except IndexError:
#             #     panorama[i][j] = 0
#             # else:
#             #     panorama[i][j] = src[vec_[0]][vec_[1]]
#                 # print("src[" + str(vec_[0]) + "][" + str(vec_[1]) +"] = " + str(src[vec_[0]][vec_[1]]))
                
#             # print("panorama[" + str(i) + "][" + str(j) + "] = " + str(panorama[i][j]))

#     # cv2.imwrite("panorama_origin.png",panorama)
    
#     return panorama

def warpPerspective(src,H,shape):
    H_inv = np.linalg.inv(H)
    panorama = np.zeros((shape[1], shape[0],3), dtype=np.uint8)

    vec_ = np.array([0,0])

    for i in range(panorama.shape[0]):
        for j in range(panorama.shape[1]):

            vec_[0] = ((H_inv[1][0]*j + H_inv[1][1]*i + H_inv[1][2])/(H_inv[2][0]*j + H_inv[2][1]*i + H_inv[2][2]))
            vec_[1] = ((H_inv[0][0]*j + H_inv[0][1]*i + H_inv[0][2])/(H_inv[2][0]*j + H_inv[2][1]*i + H_inv[2][2]))

            if vec_[0] < src.shape[0] and vec_[1] < src.shape[1] and vec_[0] >= 0 and vec_[1] >= 0:
                panorama[i][j] = src[vec_[0]][vec_[1]]
            else:
                panorama[i][j] = 0
    
    return panorama

def calculate_homography_matrix(origin, dest):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    線形DLT法にて、 変換元を変換先に対応づけるホモグラフィ行列を求める。先行実装に倣う。
    :param origin: ホモグラフィ行列計算用の初期座標配列
    :param dest: ホモグラフィ行列計算用の移動先座標配列
    :return: 計算結果のホモグラフィ行列(3 x 3)
    """

    assert origin.shape == dest.shape

    origin = __convert_corner_list_to_homography_param(origin.T)
    dest = __convert_corner_list_to_homography_param(dest.T)

    # 点を調整する（数値計算上重要）
    origin, c1 = __normalize(origin)  # 変換元
    dest, c2 = __normalize(dest)      # 変換先

    # 線形法計算のための行列を作る。
    # 点の数(nbr_correspondences)の2倍の行を持つ9列の配列
    nbr_correspondences = origin.shape[1]
    a = np.zeros((2 * nbr_correspondences, 9))

    # for：点の数だけ回す
    # 行列x(テキストのインデックスによって変更予定)の通りに値を代入
    for i in range(nbr_correspondences):
        a[2 * i] = [-origin[0][i], -origin[1][i], -1, 0, 0, 0, dest[0][i] * origin[0][i], dest[0][i] * origin[1][i],
                    dest[0][i]]
        a[2 * i + 1] = [0, 0, 0, -origin[0][i], -origin[1][i], -1, dest[1][i] * origin[0][i], dest[1][i] * origin[1][i],
                        dest[1][i]]

    # 特異値分解
    u, s, v = np.linalg.svd(a)
    # 最小二乗法の解は特異値分解した行列の最後の行
    homography_matrix = v[8].reshape((3, 3))

    # 上記調整を元に戻す
    homography_matrix = np.dot(np.linalg.inv(c2), np.dot(homography_matrix, c1))

    # 正規化(h33を1に調整)して返す
    homography_matrix = homography_matrix / homography_matrix[2, 2]

    return homography_matrix

def __normalize(point_list):
    # type: (np.ndarray) -> (np.ndarray, np.ndarray)
    """
    正規化処理
    点群を平均値が0，標準偏差が1になるように正規化
    :param point_list: 正規化対象の座標リスト
    :return: 正規化結果, 正規化に用いた係数行列(理解が怪しい)
    """
    # 平均
    m = np.mean(point_list[:2], axis=1)
    # 標準偏差
    max_std = max(np.std(point_list[:2], axis=1)) + 1e-9
    # 対角項
    c = np.diag([1 / max_std, 1 / max_std, 1])
    c[0][2] = -m[0] / max_std
    c[1][2] = -m[1] / max_std
    return np.dot(c, point_list), c


def __convert_corner_list_to_homography_param(point_list):
    # type: (np.ndarray) -> np.ndarray
    """
    点の集合（dim * n の配列）を同次座標系に変換する
    :param point_list: ホモグラフィ行列変換用に整形を行う座標リスト
    :return: ホモグラフィ変換用行列
    """
    return np.vstack((point_list, np.ones((1, point_list.shape[1]))))


def make_panorama(original1,original2):
    # 特徴点マッチング
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, False)
    # matcher.knnMatch(クエリ,トレーニング,マッチング上位いくつ返すか)
    # クエリとマッチングする上位2点ずつを返す
    matches = matcher.knnMatch(original1.des,original2.des,2)
    goodmatches = []
    trainkeys = []
    querykeys = []
    maskArray = []

    for i in matches:
        # distance ： 特徴量記述子間の距離．低いほど良い
        # 1番マッチング度合いが高い点と2番目の差が大きいければ
        if i[0].distance/i[1].distance < 0.7:
            # マッチングが高い点の組み合わせを保持
            goodmatches.append(i[0])
            querykeys.append((original1.kp[i[0].queryIdx].pt[0],original1.kp[i[0].queryIdx].pt[1]))
            trainkeys.append((original2.kp[i[0].trainIdx].pt[0],original2.kp[i[0].trainIdx].pt[1]))

    # ホモグラフィ行列の導出
    # findHomography(元平面座標,目標平面座標,導出手法, 逆投影誤差の最大値)
    # H, status = cv2.findHomography(np.array(trainkeys),np.array(querykeys),cv2.RANSAC, 5.0)
    H_ = calculate_homography_matrix(np.array(trainkeys),np.array(querykeys))

    # このT_xyの計算が無いとズレる
    # 画像のリサイズ？
    # div = calc_dst4points(H, original2.image.shape)
    # 画像合成のためにoriginal1.imageのサイズを考慮したリサイズ
    div = calc_dst4points(H_, original2.image.shape)

    d = original1.resize_mat(div)
    # 平行移動
    # 座標がマイナスにならないように
    T_xy = [[1.0, 0.0, -d[0]],[0.0, 1.0, -d[1]],[0.0, 0.0, 1.0]]
    
    # warpPerspective(元画像,変換行列，出力サイズ)

    # panorama = cv2.warpPerspective(original2.image,np.dot(T_xy,H),(original1.image.shape[1],original1.image.shape[0]))
    # panorama = cv2.warpPerspective(original2.image,np.dot(T_xy,H_),(original1.image.shape[1],original1.image.shape[0]))
    # panorama = warpPerspective(original2.image,np.dot(T_xy,H_),(original1.image.shape[1],original1.image.shape[0]))
    panorama = warpPerspective(original2.image,np.dot(T_xy,H_),(original1.image.shape[1],original1.image.shape[0]))
    # cv2.imwrite("./result/panorama_origin.png",panorama)
    # cv2.imwrite("panorama_lib.png",panorama)
    # panorama = cv2.warpPerspective(original2.image,H,(original1.image.shape[1],original1.image.shape[0]))
    CommonMask, SrcMask, TargetMask = make_mask(panorama, original1.image)
    label = cv2.connectedComponentsWithStats(CommonMask)
    center = np.delete(label[3], 0, 0)

    # マスク領域中心座標を導出
    test = get_center(CommonMask)

    # 画像の合成
    # cv2.seamlessClone(ターゲット画像，ベース画像，マスク，ターゲット画像のベース画像内での中心座標，合成時の画像処理)
    blending = cv2.seamlessClone(original1.image, panorama, cv2.cvtColor(CommonMask,cv2.COLOR_GRAY2BGR), (int(test[1]),int(test[0])), cv2.NORMAL_CLONE)
    blending = arrange_rgb(blending, TargetMask)
    blending = write_blending(blending, original1.image, SrcMask)
    return blending


def panorama_main(image_names, result_path):

    images = []
    panorama = []
    # ループ：画像の枚数分繰り返す
    for i in range(1, len(image_names)):
        print("Loading " + str(image_names[i]))
        # 画像のリサイズ
        # resize_imageはpanorama.pyの関数
        # サイズを600×800に
        # 真っ黒な画素はRGBそれぞれ+1ずつ
        img = resize_image(cv2.imread(image_names[i], cv2.IMREAD_COLOR))

        # 各画像のインスタンスを保持
        # 画像名,画像の配列,特徴点,特徴量
        # 画像表示の関数
        #
        images.append(Image(str(i), img))

    # 最初(基準)の画像を追加
    panorama.append(Image(images[0].name, images[0].image))

    print("Your images have been loaded. Generating panorama starts ...")
    for i in range(0, len(images) - 1):
        # 1枚ずつ結合する画像が増えたパノラマを順次リストに追加
        # make_panoramaはpanorama.pyの関数
        panorama.append(Image(str(i + 1), make_panorama(panorama[i], images[i + 1])))
        # cv2.imwrite("panorama"+str(i)+".png",panorama[-1].image)

    # 生成したパノラマ画像を保存
    print("A panorama image is generated.")
    cv2.imwrite(str(result_path), panorama[-1].image)


    
