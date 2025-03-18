import kagglehub
import idx2numpy
import os


def download_mnist():
    # Download latest version
    dataset_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    return dataset_path

def get_images():
    dataset_path = download_mnist()
    train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

    # Load images and labels
    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)

    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)

    return train_images, train_labels, test_images, test_labels
