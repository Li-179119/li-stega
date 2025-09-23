import cv2

# 查看cv16.png的通道数---4；查看cv16_rec_w_9000.png的通道数---3
# 读取图片
img_path = './metrics/recover100_0/cv16_rec_w_9000.png'  # 替换为你的图片路径
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# 检查图片是否读取成功
if img is None:
    print(f"Error: Unable to read {img_path}.")
else:
    # 查看图片的形状
    print(f"Image shape: {img.shape}")

    # 图片的通道数可以通过形状的第三个维度来判断
    if len(img.shape) == 3:
        channels = img.shape[2]
        print(f"Number of channels: {channels}")
    else:
        print("The image is grayscale or has no channels specified.")
