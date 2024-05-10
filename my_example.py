import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from PIL import Image
from torchvision import transforms
from modules.xfeat import XFeat


# Load and process images
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img


# Utility function to convert a tensor to a numpy array suitable for OpenCV
def tensor_to_cv2(tensor):
    # Convert tensor to numpy array
    # Adjust the channel order from CHW to HWC
    # Convert from RGB to BGR format for OpenCV
    return cv2.cvtColor(np.array(tensor.permute(1, 2, 0).mul(255).byte()), cv2.COLOR_RGB2BGR)


# Define transform for the XFeat input
transform_xfeat = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define transform for visualization (without normalization)
transform_vis = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])


# Draw keypoints on an image
def draw_keypoints(img, keypoints):
    for kpt in keypoints:
        x, y = int(kpt[0]), int(kpt[1])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Green dot
    return img


# Create plots for keypoints
def visualize(img1_kpts, img1_vis, img2_kpts, img2_vis):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(cv2.cvtColor(img1_kpts, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Image 1 Keypoints')
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(img2_kpts, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Image 2 Keypoints')
    ax[1].axis('off')

    # Create a new figure to show matches
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(np.hstack([cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB), cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB)]))
    ax.axis('off')

    # Draw lines for matches
    for (x1, y1), (x2, y2) in zip(mkpts_0, mkpts_1):
        con = ConnectionPatch(xyA=(x2 + img1_vis.shape[1], y2), xyB=(x1, y1), coordsA='data', coordsB='data',
                              axesA=ax, axesB=ax, color="yellow")
        ax.add_artist(con)

    plt.show()


if __name__ == ("__main__"):

    # Load images and process
    img2_tensor = transform_xfeat(load_image('robot_color_image.png'))
    img1_tensor = transform_xfeat(load_image('scene_matcher_rx_img.png'))

    img2_vis = tensor_to_cv2(transform_vis(load_image('robot_color_image.png')))
    img1_vis = tensor_to_cv2(transform_vis(load_image('scene_matcher_rx_img.png')))

    # Initialize XFeat
    xfeat = XFeat()

    # Detect and compute features
    output1 = xfeat.detectAndCompute(img1_tensor.unsqueeze(0), top_k = 4096)[0]
    output2 = xfeat.detectAndCompute(img2_tensor.unsqueeze(0), top_k = 4096)[0]

    # Match features
    mkpts_0, mkpts_1 = xfeat.match_xfeat(img1_tensor.unsqueeze(0), img2_tensor.unsqueeze(0))


    # Draw keypoints on images for visualization
    img1_kpts = draw_keypoints(img1_vis.copy(), output1['keypoints'])
    img2_kpts = draw_keypoints(img2_vis.copy(), output2['keypoints'])

    visualize(img1_kpts, img1_vis, img2_kpts, img2_vis)


    # Match features
    mkpts_0, mkpts_1 = xfeat.match_xfeat_star(img1_tensor.unsqueeze(0), img2_tensor.unsqueeze(0))


    # Draw keypoints on images for visualization
    img1_kpts = draw_keypoints(img1_vis.copy(), output1['keypoints'])
    img2_kpts = draw_keypoints(img2_vis.copy(), output2['keypoints'])

    visualize(img1_kpts, img1_vis, img2_kpts, img2_vis)
