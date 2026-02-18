import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

degree = 1
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
ransac = RANSACRegressor(max_trials=1000)
linear_model = make_pipeline(poly_features, ransac)


def scale_shift_align_depth(src, tar, mask, disp=False, fit_type="poly", e=1e-2):
    """
    src: HxW, 需要align的单目depth
    tar: HxW, 目标深度图, 一般是metric depth，或者sfm depth
    mask: HxW, bool mask，用于选择目标深度图中的有效区域
    disp: 是否是disparity，depth anything/midas是disparity model，需要设置为True,
            depth pro, metric3d v2是depth，需要设置为False
    fit_type: 拟合类型，可选'poly'或'ransac'
    """
    orig_src = src.copy()
    if tar.shape[0] != src.shape[0] or tar.shape[1] != src.shape[1]:
        src = cv2.resize(src, tar.shape[::-1], interpolation=cv2.INTER_NEAREST)
    if disp:
        mask = np.logical_and(np.logical_and(mask, src > 5e-4), src < 5e4)
    if mask.sum() == 0:
        print(f"No valid pixels for {fit_type}")
        raise ValueError(f"No valid pixels for {fit_type}")
    tar_val = tar[mask].astype(np.float32)
    src_val = src[mask].astype(np.float32)
    if disp:
        tar_val = np.clip(tar_val, 1e-2, None)
        tar_val = 1 / tar_val

    if fit_type == "poly":
        try:
            a, b = np.polyfit(src_val, tar_val, deg=1)
        except Exception:
            pass
            # import ipdb
            #
            # ipdb.set_trace()
    elif fit_type == "ransac":
        linear_model.fit(src_val[:, None], tar_val[:, None])
        a = linear_model.named_steps["ransacregressor"].estimator_.coef_
        b = linear_model.named_steps["ransacregressor"].estimator_.intercept_
        a, b = a.item(), b.item()
    else:
        print("Unknown fit type")
        a, b = 1, 0
    # print(f"Fit {fit_type}: scale: {a}, shift: {b}")
    if a < 0:
        print(f"Negative scale detected, {fit_type}")
        # a, b = 1, 0
        ret_tag = False
    else:
        ret_tag = True
    src_ = a * orig_src + b
    if disp:
        src_ = 1 / np.clip(src_, e, None)  # max 100 meters
    return ret_tag, src_
