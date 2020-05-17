import cv2
import dlib
import numpy as np

predictor_path = 'shape_predictor_68_face_landmarks.dat' # 模型路径

detector = dlib.get_frontal_face_detector()  # dlib的正向人脸检测器
predictor = dlib.shape_predictor(predictor_path)  # dlib的人脸形状检测器


def get_image_size(image):
    image_size = (image.shape[0], image.shape[1])
    return image_size


def get_face_landmarks(image, face_detector, shape_predictor):
    dets = face_detector(image, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found.")
        exit()
    shape = shape_predictor(image, dets[0])
    face_landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return face_landmarks


def get_face_mask(image_size, face_landmarks):
    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.concatenate([face_landmarks[0:16], face_landmarks[26:17:-1]])
    cv2.fillPoly(img=mask, pts=[points], color=255)

    # mask = np.zeros(image_size, dtype=np.uint8)
    # points = cv2.convexHull(face_landmarks)  # 凸包
    # cv2.fillConvexPoly(mask, points, color=255)
    return mask


def get_affine_image(image1, image2, face_landmarks1, face_landmarks2):
    three_points_index = [18, 8, 25]
    M = cv2.getAffineTransform(face_landmarks1[three_points_index].astype(np.float32),
                               face_landmarks2[three_points_index].astype(np.float32))
    dsize = (image2.shape[1], image2.shape[0])
    affine_image = cv2.warpAffine(image1, M, dsize)
    return affine_image.astype(np.uint8)


def get_mask_center_point(image_mask):
    image_mask_index = np.argwhere(image_mask > 0)
    miny, minx = np.min(image_mask_index, axis=0)
    maxy, maxx = np.max(image_mask_index, axis=0)
    center_point = ((maxx + minx) // 2, (maxy + miny) // 2)
    return center_point


def get_mask_union(mask1, mask2):
    mask = np.min([mask1, mask2], axis=0)  # 掩盖部分并集
    mask = ((cv2.blur(mask, (3, 3)) == 255) * 255).astype(np.uint8)  # 缩小掩模大小
    mask = cv2.blur(mask, (5, 5)).astype(np.uint8)  # 模糊掩模
    return mask


def swap(im1, face_path):
    # filename = image_face_path.split('/')[-1].split('.')[0]

    im1 = cv2.resize(im1, (600, im1.shape[0] * 600 // im1.shape[1]))
    landmarks1 = get_face_landmarks(im1, detector, predictor)  # 68_face_landmarks
    im1_size = get_image_size(im1)  # 脸图大小
    im1_mask = get_face_mask(im1_size, landmarks1)  # 脸图人脸掩模

    im2 =cv2.imread(face_path)  # camera_image
    landmarks2 = get_face_landmarks(im2, detector, predictor)  # 68_face_landmarks
    im2_size = get_image_size(im2)  # 摄像头图片大小
    im2_mask = get_face_mask(im2_size, landmarks2)  # 摄像头图片人脸掩模

    affine_im1 = get_affine_image(im1, im2, landmarks1, landmarks2)  # im1（脸图）仿射变换后的图片
    affine_im1_mask = get_affine_image(im1_mask, im2, landmarks1, landmarks2)  # im1（脸图）仿射变换后的图片的人脸掩模

    union_mask = get_mask_union(im2_mask, affine_im1_mask)  # 掩模合并
    point = get_mask_center_point(affine_im1_mask)  # im1（脸图）仿射变换后的图片的人脸掩模的中心点
    seamless_im = cv2.seamlessClone(affine_im1, im2, mask=union_mask, p=point, flags=cv2.NORMAL_CLONE)  # 进行泊松融合

    return seamless_im

