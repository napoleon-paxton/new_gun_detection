from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
from pathlib import Path
import numpy


def delete_similar_images(folder):

    FILE = Path(__file__).resolve()
    FILE = FILE.parent.parent / folder 

    print('\nTarget folder =' , FILE)

    # Getting the list of directories
    imgs_dir = os.listdir(FILE)
    
    # Checking if the list is empty or not
    if len(imgs_dir) == 0:
        print("Empty images directory... Nothing to reduce ")
        return
    else:
        print("Found images to process... ")


    imgs = FILE / '*.jpg'
    print('\nTarget folder imgs =' , str(imgs))

    # Load the OpenAI CLIP Model
    print('Loading CLIP Model...')
    model = SentenceTransformer('clip-ViT-B-32')




    # Next we compute the embeddings
    # To encode an image, you can use the following code:
    # from PIL import Image
    # encoded_image = model.encode(Image.open(filepath))
    image_names = list(glob.glob( str(imgs) ))
    print("Images:", len(image_names))
    encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

    # Now we run the clustering algorithm. This function compares images aganist 
    # all other images and returns a list with the pairs that have the highest 
    # cosine similarity score
    processed_images = util.paraphrase_mining_embeddings(encoded_image)
    NUM_SIMILAR_IMAGES = 20 

    numpy.savetxt('processed_images.txt', processed_images)


    # =================
    # DUPLICATES
    # =================
    print('Finding duplicate images...')
    # Filter list for duplicates. Results are triplets (score, image_id1, image_id2) and is scorted in decreasing order
    # A duplicate image will have a score of 1.00
    # It may be 0.9999 due to lossy image compression (.jpg)
    duplicates = [image for image in processed_images if image[0] >= 0.999]

    # Output the top X duplicate images
    # for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
    for score, image_id1, image_id2 in duplicates[:]:    
        print("\nScore: {:.3f}%".format(score * 100))
        print(image_names[image_id1])
        print(image_names[image_id2])

    # =================
    # NEAR DUPLICATES
    # =================
    print('Finding near duplicate images...')
    # Use a threshold parameter to identify two images as similar. By setting the threshold lower, 
    # you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
    # A threshold of 1.00 means the two images are exactly the same. Since we are finding near 
    # duplicate images, we can set it at 0.99 or any number 0 < X < 1.00.
    h_threshold = .99
    l_threshold = 0.985
    near_duplicates = [image for image in processed_images if l_threshold <= image[0] < h_threshold]

    for score, image_id1, image_id2 in near_duplicates[:]:
    # for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
        print("\nScore: {:.3f}%".format(score * 100))
        print(image_names[image_id1])
        print(image_names[image_id2])
        print('[INFO] ', "rm -f {}".format(image_names[image_id2]) )

        os.system("rm -f {}".format(image_names[image_id2]))

    return len(image_names), imgs