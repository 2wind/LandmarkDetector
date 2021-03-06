import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import argparse
import imutils
import pandas as pd
import numpy as np

from inference import *
from module import landmark_number, landmark_regex_string

import time
from sys import exit

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
    main_start = time.perf_counter()

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
    parser.add_argument('--output_image', type=str, help='path to the output image(verbose)')
    parser.add_argument('--debug_text', type=str, help='path to the output image(verbose)')

    args = parser.parse_args()

    model = None


    
    if (args.verbose):
        debug_logger = pd.DataFrame(columns=["Name", "parser activation", "image loading", "model loading", "landmark detection", "transform detection", "checking against answer", "image saving", "total", "diff_per_size"])

        print("\n Starting Landmark Detection...")
        print(f"NAME: {args.photo_image}")
        arg_end = time.perf_counter()
        print(f"> parser: took {arg_end - main_start:0.4f} seconds")
        debug_logger.at[0, 'Name'] = args.photo_image
        debug_logger.at[0, "parser activation"] = arg_end - main_start
    try:

        photo_image = parse_image(args.photo_image)

        
        if (args.verbose):
            img_loading_time = time.perf_counter()
            print(f"> image loading: took {img_loading_time - arg_end:0.4f} seconds")
            debug_logger.at[0, "image loading"] = img_loading_time - arg_end

        model = load_model(args.model)

        if (args.verbose):
            model_loading_time = time.perf_counter()
            print(f"> model loading: took {model_loading_time - img_loading_time:0.4f} seconds")
            debug_logger.at[0, "model loading"] = model_loading_time - img_loading_time

        photo_landmarks = find_landmark(photo_image, model)

        if (args.verbose):
            landmark_detecting_time = time.perf_counter()
            print(f"> landmark detection: took {landmark_detecting_time - model_loading_time:0.4f} seconds")
            debug_logger.at[0, "landmark detection"] = landmark_detecting_time - model_loading_time

        if (args.test):
            print(photo_landmarks)
            return

        film_landmarks = extract_landmarks(parse_tsv(args.film_tsv), landmark_regex_string, landmark_number)

        matrix = calculate_matrix(film_landmarks, photo_landmarks)
        transform = calculate_transform(film_landmarks, photo_landmarks, matrix)

        save_results(transform, args.output)
        if (args.verbose):
            transform_calc_time = time.perf_counter()
            print(f"> transform detecting: took {transform_calc_time - landmark_detecting_time:0.4f} seconds")
            debug_logger.at[0, "transform detection"] = transform_calc_time - landmark_detecting_time
            print(transform)

            if args.photo_tsv:
                
                photo_true_landmarks = extract_landmarks(parse_tsv(args.photo_tsv), landmark_regex_string, landmark_number)
                average, each = pixel_distance(photo_landmarks, photo_true_landmarks)
                print(f"average pixel difference: {average}")
                print(f"per-landmark pixel difference: ")
                print(each)
                diff_per_size = np.average(np.abs(photo_landmarks - photo_true_landmarks) / np.array(photo_image.size))
                print(f"pixel difference per image size: {diff_per_size}")
                checking_against_answer_time = time.perf_counter()
                print(f"> checking against answer: took {checking_against_answer_time - transform_calc_time:0.4f} seconds")
                debug_logger.at[0, "checking against answer"] = checking_against_answer_time - transform_calc_time
                debug_logger.at[0, "diff_per_size"] = diff_per_size
            else:
                photo_true_landmarks = None
                diff_per_size = None

            if args.output_image:
                film_image = parse_image(args.film_image)
                save_transform_image(film_landmarks, photo_landmarks, film_image, photo_image, args.output_image, matrix, photo_true_landmarks, diff_per_size)
                saving_image_time = time.perf_counter()
                print(f"> saving image: took {saving_image_time - transform_calc_time:0.4f} seconds")
                debug_logger.at[0, "image saving"] = saving_image_time - transform_calc_time

            debug_logger.at[0, "total"] = time.perf_counter() - main_start
            if (args.debug_text is not None):
                debug_logger.to_csv(args.debug_text)

    except Exception as e:
        print(e)
        parser.print_help()
    # finally:
    #     if model is not None:
    #         del model

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
    except FileNotFoundError as e:
        print(e)
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
    except FileNotFoundError as e:
        print(e)
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

def save_transform_image(film_landmarks, photo_landmarks, film_image, photo_image, photo_output, matrix, solution_landmarks=None, diff_per_size=None):
    '''
    save_transform_image(film_landmarks: np.array(),
                        photo_landmarks: np.array(), 
                        film_image: PIL.Image,
                        photo_image: PIL.Image,
                        photo_output: str,
                        matrix: np.array(),
                        solution_landmarks=None : np.array()),
                        diff_per_size:float:

    film_landmarks: landmarks in film.
    photo_landmarks: landmarks in photo.
    film_image: image of film.
    photo_image: image of photo.
    photo_output: full path to save image.
    matrix: transformation matrix from film to photo.
    solution_landmarks: landmarks in photo(solution). Optional.
    diff_per_size: difference between true landmark position and landmarks found, relative to image size. Optional.

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
        plt.text(0, photo_image.size[1], "Legend) landmarks in film(warped): red\n landmarks found in photo: green\n landmark solution of photo: blue")
    plt.text(0, -80, f"Name: {photo_output}")
    if diff_per_size is not None:
        plt.text(0, 0, f"diff: {diff_per_size}")
    plt.savefig(photo_output, bbox_inches='tight', dpi=300)
    print(f"photo saved on {photo_output}.")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"took {end - start:0.4f} seconds")