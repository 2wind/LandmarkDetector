import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import argparse
import imutils
import pandas as pd
import numpy as np

import inference
import module

parser = argparse.ArgumentParser()
# General arguments
parser.add_argument('-v', '--verbose', help='output some helpful texts', action='store_true')

# Image and tsv file arguments
parser.add_argument('-fi', '--film_image', type=str, default='film.jpg', help='path to the film image')
parser.add_argument('-ft', '--film_tsv', type=str, default='film.txt', help='path to the film landmark')
parser.add_argument('-pi', '--photo_image', type=str, default='photo.jpg', help='path to the photo image')
parser.add_argument('-pt', '--photo_tsv', type=str, help='path to the photo landmark(Optional)')

# landmark detection model arguments
parser.add_argument('-m', '--model', type=str, default='model.tar', help='path to the model')

# output arguments
parser.add_argument('-o', '--output', type=str, default='result.txt', help='path to the output')
parser.add_argument('--output_image', type=str, default='result.jpg', help='path to the output image(verbose)')

args = parser.parse_args()

def main():
    photo_image = parse_image(args.photo_image)
    film_landmarks = extract_landmarks(parse_tsv(args.film_tsv), module.landmark_regex_string, module.landmark_number)

    model = inference.load_model(args.model)

    photo_landmarks = inference.find_landmark(photo_image, model)

    matrix = inference.calculate_matrix(film_landmarks, photo_landmarks)
    transform = inference.calculate_transform(film_landmarks, photo_landmarks, matrix)

    save_results(transform, args.output)

    if (args.verbose):
        if args.photo_tsv:
            photo_true_landmarks = extract_landmarks(parse_tsv(args.photo_tsv), module.landmark_regex_string, module.landmark_number)
            average, each = inference.pixel_distance(photo_landmarks, photo_true_landmarks)
            print(f"average pixel difference: {average}")
            print(f"per-landmark pixel difference: ")
            print(each)
            film_image = parse_image(args.film_image)
            save_transform_image(film_landmarks, photo_landmarks, film_image, photo_image, args.output_image, matrix)




    del model

def parse_image(image_path: str):
    try:
        image = Image.open(image_path)
        return image
    except:
        print(f"Image not found in {image_path}")
        exit()

def parse_tsv(tsv_path: str):
    # Loading dataframe
    try:
        df = pd.read_csv(tsv_path,  sep='\t')
        df = df.iloc[:99, 0:3]
        
        df.columns = ['name', 'X', 'Y']

        return df
    except:
        print(f"Landmark text file not found in {tsv_path}")
        exit()

def extract_landmarks(df, landmark_regex, landmark_length):
    # (gathering only needed landmarks)
    df = df.loc[df['name'].str.contains(landmark_regex, regex=True), :]
    # there are **18** landmarks that is unique and valid among all files
    # should we sort df?
    df = df.sort_values(by=['name'])
    df = df.loc[:, ['X', 'Y']]
    df = df.reset_index(drop=True)

    # ... and landmark
    landmark = df.to_numpy(dtype=np.float32)
    return landmark

def save_results(transform: list, output_path:str):
    translation_x, translation_y, scale, degree = transform
    output_string = f"translation_x={translation_x}\ntranslation_y={translation_y}\nscale={scale}\ndegree={degree}\n"
    with open(output_path, "w") as output:
        output.write(output_string)


def transform_landmarks(matrix, landmarks):
    ones = np.ones((1, len(landmarks)))
    homography_landmarks = np.concatenate((landmarks, ones.T), axis=1)
    result = np.dot(matrix, homography_landmarks.T).T
    return result

def save_transform_image(film_landmarks, photo_landmarks, film_image, photo_image, photo_output, matrix):
    warped_image = cv2.warpAffine(np.array(film_image), matrix, (photo_image.size[0], photo_image.size[1]))
    warped_film = imutils.opencv2matplotlib(warped_image)
    warped_landmarks = transform_landmarks(matrix, film_landmarks)

    plt.figure()
    plt.imshow(photo_image)
    plt.imshow(warped_film, alpha=0.5)
    plt.scatter(warped_landmarks[:,0], warped_landmarks[:,1], c = 'r', s = 5)
    plt.scatter(photo_landmarks[:,0], photo_landmarks[:,1], c = 'g', s = 5)
    plt.text(0, 0, "Legend) landmarks in film(warped): red, landmarks found in photo: green")
    plt.savefig(photo_output, bbox_inches='tight')
    print(f"photo saved on {photo_output}.")

if __name__ == "__main__":
    main()