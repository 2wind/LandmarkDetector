import numpy as np
from PIL import Image
import cv2

import torch 
from torchvision.transforms.functional import to_tensor, resized_crop, normalize

from module import LandmarkNetwork, landmark_number

from facenet_pytorch import MTCNN

import time


def load_model(model_path: str):

    # set device to GPU if available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # Initialize landmark detection model using model_path.
    best_network = LandmarkNetwork(num_classes=landmark_number*2)
    # try:
    if (model_path.endswith(".tar")):
        best_network.load_state_dict(torch.load(model_path, map_location=torch.device(device))['network_state_dict'])
    else:
        best_network.load_state_dict(torch.load(model_path))
    best_network.eval()
    
    return best_network
    # except:
    #     print(f"model not found on {model_path}")
    #     del best_network
    #     exit()

def find_landmark(input_image: Image, model: LandmarkNetwork):
    '''
    find_landmark(): infers landmark from image_path, using model in model_path.
        image_path: str, full path to image inferenced.
        model: module.LandmarkNetwork(), model used to infer.

    returns: landmarks: numpy.array() in (6, 2) format, includes 6 landmarks in (x, y) format.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # start = time.perf_counter()
    # load MTCNN for face detection.
    mtcnn = MTCNN()
    
    # detect head using MTCNN
    boxes, probs = mtcnn.detect(input_image)
    face = boxes[0]
    x0, y0, x1, y1 = face
    face_width = x1 - x0
    # translate detected bounding box to right, to include nosetip to bounding box.
    x0, y0, x1, y1 = int(x0+face_width * 0.05), int(y0), int(x1 + face_width * 0.05), int(y1)

    # mtcnn_time = time.perf_counter()
    # print(f"mtcnn: took {mtcnn_time - start:0.4f} s")

    # Load grayscale image from input_image.
    grayscale_image = input_image.convert('L')
    # Crop head image, resize to 224x224, and convert it into tensor format.
    image = resized_crop(grayscale_image, y0, x0, y1-y0, x1-x0, size=(224, 224))
    image = to_tensor(image)
    image = normalize(image, [0.6945], [0.33497])

    # Detect landmarks from image.
    with torch.no_grad():
        landmarks = model(image.unsqueeze(0))

    # raw result from model is between -0.5 and 0.5.
    # convert back to original coordinate.
    landmarks = (landmarks.view(landmark_number,2).detach().numpy() + 0.5) * np.array([[x1-x0, y1-y0]]) + np.array([[x0, y0]])
    # now landmarks are in (6, 2) format.

    # inference_time = time.perf_counter()
    # print(f"inference: took {inference_time - mtcnn_time:0.4f} s")

    return landmarks

def calculate_matrix(landmark_xray, landmark_image):
    '''
    calculate_matrix(landmark_xray: numpy.array(), landmark_image: numpy.array()):
        landmark_xray: numpy.array(), landmarks in xray.
        landmark_image: numpy.array(), landmarks in image.

        Wrapper function for cv2.estimateAffinePartial2D().
        made it into own function, because sometimes certain landmarks are very unreliable.
        
        returns:  matrix: numpy.array(), a 2x3 matrix array which is affine transformation.
    '''
    matrix, inliers = cv2.estimateAffinePartial2D(landmark_xray[:], landmark_image[:], method=cv2.LMEDS)

    return matrix


def calculate_transform(landmark_xray, landmark_image, matrix):
    '''
    calculate_transform(landmark_xray: numpy.array(), landmark_image: numpy.array())
        landmark_xray: numpy.array(), landmarks in xray.
        landmark_image: numpy.array(), landmarks in image.
        matrix: numpy.array(), a 2x3 matrix array which is affine transformation.

    Calculates affine transfrom from landmark_xray to landmark_iamge.
    1. calculate partial affine transform matrix using method in opencv.
    2. derive translations, scale, and rotation(degree).
    3. return it in a list format.

    return: a list of [translation_x, translation_y, scale, degree].
    '''
    # Calculate 
    translation_x, translation_y = matrix[0, 2], matrix[1, 2]
    scale = (matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0]) ** 0.5
    rotation = np.arctan2(matrix[0, 1], matrix[1, 1])
    degree = np.rad2deg(rotation)

    return [translation_x, translation_y, scale, degree]


def pixel_distance(landmark, reference):
    ''' pixel_distance(landmark: np.array[[x, y], ..] 
                        reference: np.array[[x, y]] with true landmark value

        return: average: float average distance,
                each: np.array[distance, ..] with distance of each landmark
    '''
    each = []
    for i in range(len(landmark)):
        each.append(np.linalg.norm(landmark[i] - reference[i]))

    each = np.array(each)
    average = np.average(each)

    return average, each

