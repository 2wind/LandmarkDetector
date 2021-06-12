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

import time
from sys import exit

from facenet_pytorch import MTCNN


def main():
    '''
    main():
    
    Function that works in highest level.
    1. Opens files according to arguments and loads model
    2. Finds landmark in photo, and transformation between different landmark types.
    3. Saves transformation into text file.
    4. if verbose, saves image that contains image and landmark information.

    loads when parse.py is ran independent.
    
    returns:
        Nothing.
    '''
    parser = argparse.ArgumentParser(description="Landmark detector and transform calculator")
    # General arguments
    parser.add_argument('-v', '--verbose', help='output some helpful texts', action='store_true')
    parser.add_argument('-t', '--test', help='testing landmark detection only', action='store_true')

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

    model = None

    try:

        photo_image = parse_image(args.photo_image)

        model = inference.load_model(args.model)

        photo_landmarks = inference.find_landmark(photo_image, model)

        if (args.test):
            print(photo_landmarks)
            return

        film_landmarks = extract_landmarks(parse_tsv(args.film_tsv), module.landmark_regex_string, module.landmark_number)

        matrix = inference.calculate_matrix(film_landmarks, photo_landmarks)
        transform = inference.calculate_transform(film_landmarks, photo_landmarks, matrix)

        save_results(transform, args.output)

        if (args.verbose):
            print(transform)
            if args.photo_tsv:
                photo_true_landmarks = extract_landmarks(parse_tsv(args.photo_tsv), module.landmark_regex_string, module.landmark_number)
                average, each = inference.pixel_distance(photo_landmarks, photo_true_landmarks)
                print(f"average pixel difference: {average}")
                print(f"per-landmark pixel difference: ")
                print(each)
                diff_per_size = np.average(np.abs(photo_landmarks - photo_true_landmarks) / np.array(photo_image.size))
                print(f"pixel difference per image size: {diff_per_size}")
            else:
                photo_true_landmarks = None
                
            film_image = parse_image(args.film_image)
            save_transform_image(film_landmarks, photo_landmarks, film_image, photo_image, args.output_image, matrix, photo_true_landmarks)
    except:
        parser.print_help()
    finally:
        if model is not None:
            del model

def parse_image(image_path: str):
    '''
    parse_image(image_path: str):
        image_path: str: full path to image.

    Opens image from image_path. if fails, prints error message and quits.

    return:
        image: PIL.Image() image.
    '''
    try:
        image = Image.open(image_path)
        return image
    except FileNotFoundError:
        print(f"Image not found in {image_path}")
        exit()

def parse_tsv(tsv_path: str):
    '''
    parse_tsv(tsv_path: str):
        tsv_path: str: full path to text file, in tab separated values format.

    Opens text file in tsv_path as tsv. loads it as pandas dataframe, in a following format:

    |   name    |   X   |   Y   |
    =============================
    |   ...     |  ...  |  ...  |
    |  '29@2'   | 324.7 | 250.4 |
    |   (str)   | float | float |
    |   ...     |  ...  |  ...  |

    If fails, prints error message and quits.

    return:
        df: Pandas.dataframe() that contains landmark information.
    '''
    # Loading dataframe
    try:
        df = pd.read_csv(tsv_path,  sep='\t')
        df = df.iloc[:99, 0:3]
        
        df.columns = ['name', 'X', 'Y']

        return df
    except FileNotFoundError:
        print(f"Landmark text file not found in {tsv_path}")
        exit()

def extract_landmarks(df, landmark_regex, landmark_length):
    '''
    extract_landmarks(df: pandas.dataframe(), landmark_regex: str, landmark_length: int):
        df: Pandas.dataframe() that contains landmark information.
        landmark_regex: Regular Expression String that matches certain landmark names.
        landmark_length: length of landmarks(how many landmarks do we want to extract?).

    Gathers needed landmarks, sort, drop name, and change it into numpy array.

    returns:
        landmark: numpy.array(): (landmark_length, 2) shaped numpy array with landmark(x, y) in each row.
    '''
    # (gathering only needed landmarks)
    df = df.loc[df['name'].str.contains(landmark_regex, regex=True), :]
    # there are **18** landmarks that is unique and valid among all files
    df = df.sort_values(by=['name'])
    df = df.loc[:, ['X', 'Y']]
    df = df.reset_index(drop=True)

    # ... and landmark
    landmark = df.to_numpy(dtype=np.float32)
    return landmark

def save_results(transform: list, output_path:str):
    '''
    save_results(transform: list, output_path:str):
        transform: a list of [translation_x, translation_y, scale, degree].
        output_path: path to save output text file.

    saves tranform into text file as defined in output_string.
    file is saved to output_path.

    returns:
        nothing.
    '''
    translation_x, translation_y, scale, degree = transform
    output_string = f"translation_x={translation_x}\ntranslation_y={translation_y}\nscale={scale}\ndegree={degree}\n"
    with open(output_path, "w") as output:
        output.write(output_string)
    print(f"transform output saved to {output_path}")


def transform_landmarks(matrix, landmarks):
    '''
    transform_landmarks(matrix: np.array(), landmarks: np.array()):
        matrix: numpy.array(), a 2x3 matrix array which is affine transformation.
        landmarks: numpy.array(), a (n, 2) shaped array that contains landmarks information.

    1. Add "1" to each coordinate. (x, y) --> (x, y, 1)
    2. Transpose and multiply with matrix.
        [[a, b, c],   [[x,         [[x',
         [d, e, f]] X   y,      =    y'], ...]
                        1], ...] 
            matrix  X landmarks = transformed matrix
    3. Transpose back to original format, and return.

    returns:
        result: numpy.array(), a (n, 2) shaped array that contains transformed landmarks information.
        
    '''
    ones = np.ones((1, len(landmarks)))
    homography_landmarks = np.concatenate((landmarks, ones.T), axis=1)
    result = np.dot(matrix, homography_landmarks.T).T
    return result

def save_transform_image(film_landmarks, photo_landmarks, film_image, photo_image, photo_output, matrix, solution_landmarks=None):
    '''
    save_transform_image(film_landmarks: np.array(),
                        photo_landmarks: np.array(), 
                        film_image: PIL.Image,
                        photo_image: PIL.Image,
                        photo_output: str,
                        matrix: np.array(),
                        solution_landmarks=None : np.array()):

    film_landmarks: landmarks in film.
    photo_landmarks: landmarks in photo.
    film_image: image of film.
    photo_image: image of photo.
    photo_output: full path to save image.
    matrix: transformation matrix from film to photo.
    solution_landmarks: landmarks in photo(solution). Optional.

    Saves image that contains photo, film image(warped), with landmark dots overlayed.

    1. Warps film image using cv2.warpAffine() function and matrix, and convert back to PIL.Image().
    2. Transform landmark coordinates in film_landmarks using matrix.
    3. Initialize plt figure, and add photo_image.
    4. Add warped film image to figure, with 0.5 alpha(50% transparency)
    5. Add dots from warped film landmark, photo landmarks, and if exists, soultion landmarks for photo.
    6. Add simple legend at the top of the image.
    7. Save figure to photo_output.

    returns:
        Nothing.
    '''

    warped_image = cv2.warpAffine(np.array(film_image), matrix, (photo_image.size[0], photo_image.size[1]))
    warped_film = imutils.opencv2matplotlib(warped_image)
    warped_landmarks = transform_landmarks(matrix, film_landmarks)

    plt.figure()
    plt.imshow(photo_image)
    plt.imshow(warped_film, alpha=0.5)
    plt.scatter(warped_landmarks[:,0], warped_landmarks[:,1], c = 'r', s = 5)
    plt.scatter(photo_landmarks[:,0], photo_landmarks[:,1], c = 'g', s = 5)
    if (solution_landmarks is not None):
        plt.scatter(solution_landmarks[:,0], solution_landmarks[:,1], c = 'b', s=5)
    plt.text(0, 0, "Legend) landmarks in film(warped): red\n landmarks found in photo: green\n landmark solution of photo: blue")
    plt.savefig(photo_output, bbox_inches='tight', dpi=300)
    print(f"photo saved on {photo_output}.")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"took {end - start:0.4f} seconds")