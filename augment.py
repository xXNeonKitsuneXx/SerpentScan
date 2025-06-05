import os
import cv2
import albumentations as A

def augment_images(input_root, output_root, n):
    # Define safe augmentations
    transform = A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.6),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.9),

        # Texture, blur, and noise
        A.MedianBlur(blur_limit=3, p=0.4),
        A.GaussNoise(p=0.2),

        # Exposure and tone
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.RandomToneCurve(scale=0.1, p=0.2),

        # Weather & lighting simulation
        A.RandomRain(blur_value=3, brightness_coefficient=0.9, drop_width=1, drop_length=20, p=0.2),
        A.RandomSnow(brightness_coeff=1.2, p=0.2),
        A.RandomFog(alpha_coef=0.08, p=0.2),
        A.RandomShadow(shadow_dimension=5, p=0.2),
        A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5), src_radius=200, src_color=(255, 255, 255), p=0.1),

        # Occlusion
        A.CoarseDropout(p=0.2),
    ])
    # Loop through each class folder
    for class_name in os.listdir(input_root):
        class_input_dir = os.path.join(input_root, class_name)

        # Skip files like .DS_Store
        if not os.path.isdir(class_input_dir):
            continue

        class_output_dir = os.path.join(output_root, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        for filename in os.listdir(class_input_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            filepath = os.path.join(class_input_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f"Warning: Could not read {filepath}")
                continue

            base_filename = os.path.splitext(filename)[0]
            for i in range(n):
                augmented = transform(image=image)['image']
                output_filename = f"{base_filename}_aug_{i+1}.jpg"
                cv2.imwrite(os.path.join(class_output_dir, output_filename), augmented)

    print(f"✅ All augmented images saved to: {output_root}")


if __name__ == "__main__":
    augment_images("snake_images", "snake_images_augment", n=10)