import cv2
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def white_balance(image):
    """
    Apply automatic white balance to the image using the gray world algorithm
    
    Args:
        image (numpy.ndarray): The input image in BGR format
        
    Returns:
        numpy.ndarray: The white-balanced image
    """
    try:
        # 使用简单的灰度世界算法实现白平衡
        # 计算每个通道的平均值
        b, g, r = cv2.split(image)
        r_avg = np.mean(r)
        g_avg = np.mean(g)
        b_avg = np.mean(b)
        
        # 计算亮度的平均值 (灰度世界假设)
        k = (r_avg + g_avg + b_avg) / 3
        
        # 计算增益系数
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg
        
        # 应用增益调整
        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        
        # 合并通道
        balanced_image = cv2.merge([b, g, r])
        return balanced_image
    except Exception as e:
        logger.warning(f"White balance failed: {e}")
        return image

def light_normalization(image):
    """
    Apply light normalization to the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image (numpy.ndarray): The input image in BGR format
        
    Returns:
        numpy.ndarray: The normalized image
    """
    try:
        # Convert image to Lab color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L channel back with the a and b channels
        limg = cv2.merge((cl, a, b))
        image_normalized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return image_normalized
    except Exception as e:
        logger.warning(f"Light normalization failed: {e}")
        return image

def color_correction(image):
    """
    Apply color correction to the image
    
    Args:
        image (numpy.ndarray): The input image in BGR format
        
    Returns:
        numpy.ndarray: The color-corrected image
    """
    try:
        # 使用最大值法进行色彩校正
        b, g, r = cv2.split(image)
        
        # 找出图像中最亮的点作为参考白点
        max_b = np.max(b)
        max_g = np.max(g)
        max_r = np.max(r)
        
        # 根据最大值调整通道
        if max_b > 0 and max_g > 0 and max_r > 0:
            b = cv2.multiply(b, 255.0 / max_b)
            g = cv2.multiply(g, 255.0 / max_g)
            r = cv2.multiply(r, 255.0 / max_r)
        
        # 合并通道
        corrected_image = cv2.merge([b, g, r])
        return corrected_image.astype(np.uint8)
    except Exception as e:
        logger.warning(f"Color correction failed: {e}")
        return image

def retinex_enhancement(image):
    """
    Apply Retinex enhancement to remove shadows and enhance details
    
    Args:
        image (numpy.ndarray): The input image in BGR format
        
    Returns:
        numpy.ndarray: The enhanced image
    """
    try:
        # 实现简化版的单尺度Retinex算法
        # 转换为YUV空间
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        
        # 对亮度通道应用高斯模糊
        blur = cv2.GaussianBlur(y, (0, 0), 10)
        
        # 避免除零问题
        blur = np.where(blur == 0, 0.1, blur)
        
        # 计算亮度通道与模糊版本的对数差异(Retinex理论)
        y_retinex = cv2.log(y.astype(np.float32) + 1.0) - cv2.log(blur.astype(np.float32) + 1.0)
        
        # 归一化并放大对比度
        y_retinex = 255.0 * (y_retinex - np.min(y_retinex)) / (np.max(y_retinex) - np.min(y_retinex) + 0.0001)
        
        # 转回uint8
        enhanced_y = y_retinex.astype(np.uint8)
        
        # 合并回YUV并转回BGR
        enhanced_yuv = cv2.merge([enhanced_y, u, v])
        enhanced_image = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)
        
        return enhanced_image
    except Exception as e:
        logger.warning(f"Retinex enhancement failed: {e}")
        return image

def gamma_correction(image, gamma=1.2):
    """
    Apply gamma correction to adjust image brightness
    
    Args:
        image (numpy.ndarray): The input image in BGR format
        gamma (float): Gamma value for correction (default: 1.2)
        
    Returns:
        numpy.ndarray: The gamma-corrected image
    """
    try:
        # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    except Exception as e:
        logger.warning(f"Gamma correction failed: {e}")
        return image

def denoise_image(image, strength=10):
    """
    Apply denoising to the image
    
    Args:
        image (numpy.ndarray): The input image in BGR format
        strength (int): Strength of denoising (default: 10)
        
    Returns:
        numpy.ndarray: The denoised image
    """
    try:
        # Use Non-Local Means Denoising algorithm
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        return denoised_image
    except Exception as e:
        logger.warning(f"Denoising failed: {e}")
        return image

def contrast_enhancement(image, alpha=1.2, beta=10):
    """
    Enhance the contrast of the image
    
    Args:
        image (numpy.ndarray): The input image in BGR format
        alpha (float): Contrast control (default: 1.2)
        beta (int): Brightness control (default: 10)
        
    Returns:
        numpy.ndarray: The contrast-enhanced image
    """
    try:
        # Apply the formula: new_image = alpha * old_image + beta
        enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced_image
    except Exception as e:
        logger.warning(f"Contrast enhancement failed: {e}")
        return image

def preprocess_image(image):
    """
    Apply a sequence of preprocessing steps to enhance the tongue image
    
    Args:
        image (numpy.ndarray): The input image in BGR format
        
    Returns:
        numpy.ndarray: The preprocessed image
    """
    try:
        # Step 1: Denoise the image
        image = denoise_image(image, strength=5)
        
        # Step 2: Apply automatic white balance
        image = white_balance(image)
        
        # Step 3: Apply light normalization
        image = light_normalization(image)
        
        # Step 4: Apply color correction
        image = color_correction(image)
        
        # Step 5: Apply Retinex enhancement for shadow removal and detail enhancement
        image = retinex_enhancement(image)
        
        # Step 6: Apply gamma correction
        image = gamma_correction(image, gamma=1.2)
        
        # Step 7: Enhance contrast
        image = contrast_enhancement(image, alpha=1.1, beta=5)
        
        return image
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        # If preprocessing fails, return the original image
        return image

def load_and_preprocess_image(image_path):
    """
    Load an image from the file system and apply preprocessing
    
    Args:
        image_path (str or Path): Path to the image file
        
    Returns:
        numpy.ndarray: The preprocessed image or None if loading fails
    """
    try:
        # Read image from file
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        
        # Apply preprocessing
        preprocessed_image = preprocess_image(image)
        
        return preprocessed_image
    except Exception as e:
        logger.error(f"Error loading and preprocessing image {image_path}: {e}")
        return None 