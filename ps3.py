"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    h = image.shape[0]
    w = image.shape[1]
    return [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    img = image.copy()
    # img_medianblur = cv2.medianBlur(img, 9)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.medianBlur(gray, 9)
    ksize_gaussian = 11
    sigmaX_gaussian = 3
    img_blur = cv2.GaussianBlur(gray, (ksize_gaussian, ksize_gaussian), sigmaX_gaussian)

    # cv2.imshow('corners', img_blur)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    img_blur = np.float32(img_blur)
    # dst = cv2.cornerHarris(img_blur, blockSize=2, ksize=3, k=0.04)  # orig
    # dst = cv2.cornerHarris(img_blur, blockSize=12, ksize=11, k=0.18)  # pass one
    dst = cv2.cornerHarris(img_blur, blockSize=12, ksize=9, k=0.18)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    corners = np.array(np.where(dst > 0.01 * dst.max())).T
    corners = np.float32(corners)
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(corners, 4, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = []
    # print center
    for row in center:
        centers.append([row[1], row[0]])
    # sort by x
    sorted_centers = []
    centers.sort(key=lambda t: t[0])
    # sort by y
    if centers[0][1] > centers[1][1]:
        sorted_centers.append(centers[1])
        sorted_centers.append(centers[0])
    else:
        sorted_centers.append(centers[0])
        sorted_centers.append(centers[1])
    if centers[2][1] > centers[3][1]:
        sorted_centers.append(centers[3])
        sorted_centers.append(centers[2])
    else:
        sorted_centers.append(centers[2])
        sorted_centers.append(centers[3])

    # Now draw them
    # res = np.hstack((centroids, corners))
    # res = np.int0(res)
    # img[res[:,1],res[:,0]]=[0,0,255]
    # img[res[:,3],res[:,2]] = [0,255,0]
    # Threshold for an optimal value, it may vary depending on the image.
    # y = img[dst > 0.01 * dst.max()][1]
    # print y
    # img[dst > 0.01 * dst.max()] = [0, 0, 255]
    # # dst = np.uint8(dst)
    # cv2.imshow('corners', img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    sorted_centers = np.uint(sorted_centers)
    return [tuple(sorted_centers[0]), tuple(sorted_centers[1]), tuple(sorted_centers[2]), tuple(sorted_centers[3])]


def draw_box(image, markers, thickness=3):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    img = image.copy()
    color = (0, 50, 255)
    cv2.line(img, markers[0], markers[1], color=color, thickness=thickness)
    cv2.line(img, markers[1], markers[3], color=color, thickness=thickness)
    cv2.line(img, markers[3], markers[2], color=color, thickness=thickness)
    cv2.line(img, markers[2], markers[0], color=color, thickness=thickness)
    return img


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    # create a output image as the same size with imageB
    img = imageB.copy()
    # make a big ju zhen like [ [suo you xd], [suo you yd], [dou shi 1]] store all destination zuo biao
    height, width, cb = imageB.shape
    indices_dst = np.empty([height * width, 3])
    # zuo biao de ji he xiang zhe yang:
    # [[0,1,2,3,4,...,width, 0,1,2,3,4,...,width,0,1,2,3,4...,width], yi gong chong fu height bian
    #  [0,0,0...yi gong width ge ling, 1,1,1,...yi gong width ge 1, ...zhi dao width ge height-1]
    #  [1,1,1,...yi gong height X width ge yi]]
    indices_dst[:, 0] = np.tile(np.arange(width), height)
    indices_dst[:, 1] = np.repeat(np.arange(height), width)
    indices_dst[:, 2] = 1
    # calculate the soure image de zuo biao da ji he
    h_inv = np.linalg.inv(homography)
    indices_src_wT = np.dot(h_inv, indices_dst.T)
    # cancel the 'w' and the last column
    indices_src = (indices_src_wT / indices_src_wT[-1]).T[:, 0:2]
    # indices_src = (indices_src_wT / indices_src_wT[-1]).T   # for vectorization

    # ba ju zhen tan ping hao cao zuo
    img = np.reshape(img, (height * width, 3))
    imgA_boudary_y, imgA_boudary_x, imgA_c = imageA.shape

    # print img.shape
    # print indices_src.shape
    # try vectorization not working
    # img[0 <= indices_src[:, 0] < imgA_boudary_x and 0 <= indices_src[:, 1] < imgA_boudary_y] = imageA[np.int(indices_src[:, 1])][np.int(indices_src[:, 0])]

    # zhi jie yi dui yi fu zhi
    for i in range(img.shape[0]):
        if 0 <= indices_src[i][0] < imgA_boudary_x and 0 <= indices_src[i][1] < imgA_boudary_y:
            img[i] = imageA[np.int(indices_src[i][1])][np.int(indices_src[i][0])]

    # yong ping jun zhi fu zhi not working
    # for i in range(img.shape[0]):
    #     if 0 <= indices_src[i][0] < imgA_boudary_x-2 and 0 <= indices_src[i][1] < imgA_boudary_y-2:
    #         if 0.25 < indices_src[i][1] - np.int(indices_src[i][1]) < 0.75:
    #             img[i] = imageA[np.int(indices_src[i][1])][np.int(indices_src[i][0])]/2 + imageA[np.int(indices_src[i][1])][np.int(indices_src[i+1][0])]/2
    #         else:
    #             img[i] = imageA[np.int(indices_src[i][1])][np.int(indices_src[i][0])]
    # cao zuo wan zai bian hui yuan lai xing zhuang
    return img.reshape(height, width, 3)


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    H = np.zeros((3, 3))
    A = np.zeros((8, 8))

    A[0, :] = [src_points[0][0], src_points[0][1], 1, 0, 0, 0, -(src_points[0][0] * dst_points[0][0]), -(src_points[0][1] * dst_points[0][0])]
    A[1, :] = [0, 0, 0, src_points[0][0], src_points[0][1], 1, -(src_points[0][0] * dst_points[0][1]), -(src_points[0][1] * dst_points[0][1])]
    A[2, :] = [src_points[1][0], src_points[1][1], 1, 0, 0, 0, -(src_points[1][0] * dst_points[1][0]), -(src_points[1][1] * dst_points[1][0])]
    A[3, :] = [0, 0, 0, src_points[1][0], src_points[1][1], 1, -(src_points[1][0] * dst_points[1][1]), -(src_points[1][1] * dst_points[1][1])]
    A[4, :] = [src_points[2][0], src_points[2][1], 1, 0, 0, 0, -(src_points[2][0] * dst_points[2][0]), -(src_points[2][1] * dst_points[2][0])]
    A[5, :] = [0, 0, 0, src_points[2][0], src_points[2][1], 1, -(src_points[2][0] * dst_points[2][1]), -(src_points[2][1] * dst_points[2][1])]
    A[6, :] = [src_points[3][0], src_points[3][1], 1, 0, 0, 0, -(src_points[3][0] * dst_points[3][0]), -(src_points[3][1] * dst_points[3][0])]
    A[7, :] = [0, 0, 0, src_points[3][0], src_points[3][1], 1, -(src_points[3][0] * dst_points[3][1]), -(src_points[3][1] * dst_points[3][1])]

    b = np.asarray([dst_points[0][0], dst_points[0][1], dst_points[1][0], dst_points[1][1],
                    dst_points[2][0], dst_points[2][1], dst_points[3][0], dst_points[3][1]])

    x = np.linalg.solve(A, b)

    H[0, 0] = x[0]
    H[0, 1] = x[1]
    H[0, 2] = x[2]
    H[1, 0] = x[3]
    H[1, 1] = x[4]
    H[1, 2] = x[5]
    H[2, 0] = x[6]
    H[2, 1] = x[7]
    H[2, 2] = 1

    return H


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)
    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None
