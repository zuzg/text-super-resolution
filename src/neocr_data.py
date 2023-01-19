import cv2
from bs4 import BeautifulSoup
import glob
import random
import numpy as np
from src.utils import imshow

def get_text_points_list(path:str) -> list[tuple]:
    with open(path, 'r', encoding="utf8") as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, "xml")

    objects_all_info = Bs_data.find_all('object')
    text_polygons = [(obj.find('text').text, obj.find('polygon')) for obj in objects_all_info if obj.find('text') is not None and obj.findChild('polygon') is not None]
    text_points = list()
    for text, polygon in text_polygons:
        points = list()
        for x, y in zip(polygon.find_all('x'), polygon.find_all('y')):
            points.append((int(x.text), int(y.text)))
        text_points.append((text,points))
    return text_points

def get_text_images_from_img(img_path:str, xml_path:str, display:bool=False) -> list[tuple[np.ndarray]]:
    image = cv2.imread(img_path, 1)
    # print(image.shape)
    if display:
        imshow(cv2.resize(image, None, fx=0.25, fy=0.25))
    text_points = get_text_points_list(xml_path)

    images = list()
    for text, points in text_points:
        x, y = min(points)
        x2, y2 = max(points)

        if display:
            print(text)
            print(f'(x1, y1): {x, y},\t(x2, y2): {x2, y2}')

        try:
            image_cropped = image[y:y2, x:x2, :]
            if y2-y > x2-x:
                image_cropped = cv2.rotate(image_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_cropped = cv2.resize(image_cropped, (128, 32))
        except Exception as e:
            # print(str(e))
            continue

        image_cropped_LR = cv2.resize(cv2.resize(image_cropped, (50, 10)), (128,32))

        if display:
            imshow(image_cropped)
            imshow(image_cropped_LR)
        images.append((image_cropped_LR,image_cropped))
    return images

def get_neocr_images(n:int=None, shuffle:bool=False) -> list[tuple[np.ndarray]]:    
    images = list()
    image_files = glob.glob("data/neocr_dataset/Images/users/pixtract/dataset/*.jpg")
    if shuffle:
        random.shuffle(image_files)
    for img_path in image_files:
        img_path = img_path.replace("\\","/")
        xml_path = img_path.replace("Images","Annotations").replace("jpg", "xml")
        # print(img_path)
        # print(xml_path)
        images.extend(get_text_images_from_img(img_path, xml_path))
        if n is not None and len(images) >= n:
            return images[:n]
    return images 
    