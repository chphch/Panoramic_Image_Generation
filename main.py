import cv2
import numpy as np
from scipy.interpolate import interpn
from sklearn.neighbors import NearestNeighbors


def extractSIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    loc = np.array([k.pt for k in kp])
    return loc, des


def MatchSIFT(loc1, des1, loc2, des2, filter_ratio=0.5):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """

    # Find matching indices
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(des2)
    neigh_dist_1_img2, neigh_ind_1_img2 = neigh.kneighbors(des1, return_distance=True) # (n1,?)
    neigh.fit(des1)
    neigh_dist_2_img1, neigh_ind_2_img1 = neigh.kneighbors(des2, return_distance=True) # (n2,?)

    filtered_bool_1_1to2 = neigh_dist_1_img2[:, 0] < filter_ratio * neigh_dist_1_img2[:, 1]
    filtered_bool_2_2to1 = neigh_dist_2_img1[:, 0] < filter_ratio * neigh_dist_2_img1[:, 1]
    filtered_ind_2_1to2 = neigh_ind_1_img2[filtered_bool_1_1to2, 0]
    filtered_bool_2_bi = np.in1d(np.arange(loc2.shape[0]), filtered_ind_2_1to2) & filtered_bool_2_2to1
    filtered_ind_2_bi = np.where(filtered_bool_2_bi)[0]
    filtered_ind_1_bi = neigh_ind_2_img1[filtered_ind_2_bi, 0]

    x1 = loc1[filtered_ind_1_bi]
    x2 = loc2[filtered_ind_2_bi]
    return x1, x2


def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC
    
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    n = x1.shape[0]
    A_parts = []
    for x1_point, x2_point in zip(x1, x2):
        x = x1_point[0]
        y = x1_point[1]
        x_ = x2_point[0]
        y_ = x2_point[1]
        A_parts.append(np.array(
            [[x, y, 1, 0, 0, 0, -x_ * x, -x_ * y, -x_],
            [0, 0, 0, x, y, 1, -y_ * x, -y_ * y, -y_]]))

    x1_homo_true = np.hstack([x1, np.ones(n).reshape(-1, 1)])
    x2_homo_true = np.hstack([x2, np.ones(n).reshape(-1, 1)])

    H_best = None
    inlier_best = np.array([])
    for _ in range(ransac_n_iter):
        indices = np.random.choice(n, 4, replace=False)
        A = np.vstack([A_parts[index] for index in indices])
        U, S, VT = np.linalg.svd(A)
        H = VT[-1].reshape(3, 3)
        x2_homo_pred = (H @ x1_homo_true.T).T
        x2_homo_pred_scaled = x2_homo_pred / x2_homo_pred[:, [2]]
        distances = np.linalg.norm(x2_homo_true[:, :2] - x2_homo_pred_scaled[:, :2], axis=1)
        inlier = np.where(distances < ransac_thr)[0]
        if inlier_best.shape[0] < inlier.shape[0]:
            H_best = H
            inlier_best = inlier
    return H_best, inlier


def EstimateR(H, K):
    """
    Compute the relative rotation matrix
    
    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """
    
    R = np.linalg.inv(K) @ H @ K
    U, S, VT = np.linalg.svd(R)
    R_temp = U @ VT
    R = np.sign(np.linalg.det(R_temp)) * R_temp
    return R


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface
    
    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """

    f = K[0, 0]
    hh, ww = np.meshgrid(np.arange(Hc), np.arange(Wc), indexing='ij')
    pi = 2 * np.pi * ww / Wc
    P = np.stack([f * np.sin(pi), hh - Hc / 2, f * np.cos(pi)], axis=2)
    return P


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane
    
    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """
    
    p_rotated = np.einsum('mn,hwn->hwm', R, p)
    p_projected = np.einsum('mn,hwn->hwm', K, p_rotated)
    u = (p_projected / p_projected[:, :, [2]])[:, :, :2]
    u_x = u[:, :, 0]
    u_y = u[:, :, 1]
    p_rotated_z = p_rotated[:, :, 2]
    mask = (u_x >= 0) & (u_x < W) & (u_y >= 0) & (u_y < H) & (p_rotated_z >= 0)
    return u, mask


def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas
    
    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """
    
    hrange = np.arange(image_i.shape[0])
    wrange = np.arange(image_i.shape[1])
    u_yx = u[:, :, [1, 0]]
    canvas_i = interpn((hrange, wrange), image_i, u_yx, bounds_error=False)
    mask_i_3d = np.stack([mask_i for _ in range(3)], axis=-1)
    canvas_i *= mask_i_3d
    return canvas_i


def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image
    
    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """

    canvas[mask_i] = canvas_i[mask_i]
    return canvas


if __name__ == '__main__':
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([
        [320, 0, 480],
        [0, 320, 270],
        [0, 0, 1]
    ])

    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = '{}.jpg'.format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = [np.eye(3)]
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
        img1 = im_list[i]
        img2 = im_list[i + 1]
		
        # Extract SIFT features
        loc1, des1 = extractSIFT(img1)
        loc2, des2 = extractSIFT(img2)
		
        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2, filter_ratio=0.7)

        # Draw SIFT matching
        img_SIFT_matched = np.hstack([img1, img2])
        img2_x_offset = img1.shape[1]
        for (xp1_x, xp1_y), (xp2_x, xp2_y) in zip(x1, x2):
            color = np.random.randint(256, size=3, dtype=int).tolist()
            cv2.line(img_SIFT_matched, (int(xp1_x), int(xp1_y)),
                (int(img2_x_offset + xp2_x), int(xp2_y)), color, 1)
        cv2.imwrite(f'SIFT_match_{i}.jpg', cv2.cvtColor(img_SIFT_matched, cv2.COLOR_RGB2BGR))

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)

		# Compute R_new (or R_i+1)
        R_new = rot_list[-1] @ R
        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]
    
    Hc = Him
    Wc = len(im_list) * Wim // 2
	
    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):

        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)

        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)

        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)

        # Save the canvas
        cv2.imwrite(f'output_{i + 1}.png', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
