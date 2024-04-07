import json
from PIL import Image, ImageDraw
import os
import pandas as pd


def path_finder(path):
    """
    Retrieves the path of files in differents data folders
    """
    train_dir = path
    paths = []

    for i in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, i)
        if i != '.DS_Store':
            for j in os.listdir(folder_path):
                file_path = os.path.join(folder_path, j)
                paths.append(file_path)
    return paths


def path_finder_one_folder(path):
    """
    Retrieves the path of files in one data folders
    """
    dir = path
    paths = []

    for i in os.listdir(dir):
        folder_path = os.path.join(dir, i)
        if i != '.DS_Store':
            paths.append(folder_path)
    return paths


def create_dataframe(image_paths, label_paths):
    """
    Create and cleaning Dataframe
    """
    df_image_paths = pd.DataFrame({"Index": "", "Image_Path": image_paths})
    df_label_paths = pd.DataFrame({"Index": "", "Target_Path": label_paths})

    filter_instanceIds = df_label_paths[df_label_paths['Target_Path']
                                        .str.contains('instanceIds')]
    filter_color = df_label_paths[df_label_paths['Target_Path']
                                  .str.contains('color')]
    filter_labelIds = df_label_paths[df_label_paths['Target_Path']
                                     .str.contains('labelIds')]

    df_label_paths = df_label_paths.drop(filter_instanceIds.index)
    df_label_paths = df_label_paths.drop(filter_color.index)
    df_label_paths = df_label_paths.drop(filter_labelIds.index)

    index_tab = []
    for el in df_image_paths['Image_Path']:
        file_name = os.path.splitext(os.path.basename(el))[0]
        file_name = file_name.replace("_leftImg8bit", "")
        index_tab.append(file_name)
    df_image_paths['Index'] = index_tab

    index_tab = []
    for el in df_label_paths['Target_Path']:
        file_name = os.path.splitext(os.path.basename(el))[0]
        file_name = file_name.replace("_gtFine_polygons", "")
        index_tab.append(file_name)
    df_label_paths['Index'] = index_tab

    df = pd.merge(df_image_paths, df_label_paths, on='Index', how='left')
    return df


def mask_transformation_saving(df, path=""):
    """
    From JSON data, drawing of masks with choice of
    colors and saving in a dedicated folder
    """
    for i in range(len(df)):
        path_JSON = df['Target_Path'][i]
        file_path = df['Index'][i]
        with open(path_JSON, 'r') as f:
            json_file = json.load(f)
        img = Image.new("RGB", (json_file['imgWidth'],
                                json_file['imgHeight']), "black")
        draw = ImageDraw.Draw(img)

        for object in json_file['objects']:
            polygon_annotation = []
            """ VEHICLE """
            if object['label'] == 'car':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(250, 170, 30, 255))
            elif object['label'] == 'truck':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(250, 170, 30, 255))
            elif object['label'] == 'bus':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(250, 170, 30, 255))
            elif object['label'] == 'on rails':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(250, 170, 30, 255))
            elif object['label'] == 'motorcycle':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(250, 170, 30, 255))
            elif object['label'] == 'bicycle':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(250, 170, 30, 255))
            elif object['label'] == 'caravan':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(250, 170, 30, 255))
            elif object['label'] == 'trailer':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(250, 170, 30, 255))
                """ FLAT """
            elif object['label'] == 'road':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(0, 0, 142, 255))
            elif object['label'] == 'sidewalk':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(0, 0, 142, 255))
            elif object['label'] == 'parking':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(0, 0, 142, 255))
            elif object['label'] == 'rail track':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(0, 0, 142, 255))
                """ HUMAN """
            elif object['label'] == 'person':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(102, 102, 156, 255))
            elif object['label'] == 'rider':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(102, 102, 156, 255))
                """ CONSTRUCTION """
            elif object['label'] == 'building':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(220, 20, 60, 255))
            elif object['label'] == 'wall':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(220, 20, 60, 255))
            elif object['label'] == 'fence':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(220, 20, 60, 255))
            elif object['label'] == 'guard rail':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(220, 20, 60, 255))
            elif object['label'] == 'bridge':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(220, 20, 60, 255))
            elif object['label'] == 'tunnel':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(220, 20, 60, 255))
                """ OBJECT """
            elif object['label'] == 'pole':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(153, 153, 153, 255))
            elif object['label'] == 'pole group':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(153, 153, 153, 255))
            elif object['label'] == 'traffic sign':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(153, 153, 153, 255))
            elif object['label'] == 'traffic light':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(153, 153, 153, 255))
                """ NATURE """
            elif object['label'] == 'vegetation':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(244, 35, 232, 255))
            elif object['label'] == 'terrain':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(244, 35, 232, 255))
                """ SKY """
            elif object['label'] == 'sky':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(70, 70, 70, 255))
                """ VOID """
            elif object['label'] == 'ground':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(70, 130, 180, 255))
            elif object['label'] == 'dynamic':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(70, 130, 180, 255))
            elif object['label'] == 'static':
                for x, y in object['polygon']:
                    polygon_annotation.append((x, y))
                draw.polygon(polygon_annotation, fill=(70, 130, 180, 255))
            else:
                pass
        img.save(f'./Data/masks{path}/{file_path}.png')


def saving_df(df, path):
    dataset = path
    df.to_csv(dataset, index=False)


def main():
    image_paths_train = path_finder(
        "./Data/files/Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/train")
    label_paths_train = path_finder(
        "./Data/files/Cityscapes_gtFine_trainvaltest/gtFine/train")
    image_paths_val = path_finder(
        "./Data/files/Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/val")
    label_paths_val = path_finder(
        "./Data/files/Cityscapes_gtFine_trainvaltest/gtFine/val")

    df_train = create_dataframe(image_paths_train, label_paths_train)
    df_val = create_dataframe(image_paths_val, label_paths_val)

    mask_transformation_saving(df_train)
    mask_transformation_saving(df_val, "_val")

    image_paths_train = path_finder(
        "./Data/files/Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/train")
    label_paths_train = path_finder_one_folder('./Data/masks')
    image_paths_val = path_finder(
        "./Data/files/Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/val")
    label_paths_val = path_finder_one_folder('./Data/masks_val')

    df_train = create_dataframe(image_paths_train, label_paths_train)
    df_val = create_dataframe(image_paths_val, label_paths_val)

    saving_df(df_train, "./Data/df_train.csv")
    saving_df(df_val, "./Data/df_val.csv")


if __name__ == "__main__":
    main()
