from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from predict_utils import *


def pca(batch_size, vgg_layer, component=None):
    '''
    Takes a VGG16 layer and returns a transformed PCA version.
    
    TODO: finalize
    
    Parameters:
        - batch_size: int representing the total number of images
        - vgg_layer: numpy array representing a vgg16 layer
        - component: int representing the component to use
    
    Returns:
        - X_pca: numpy array 
    '''
    X = vgg_layer.reshape((batch_size, -1))

    pca = PCA(n_components=component) # n_components = min(component, n_samples)
    pca.fit(X)
    X_pca = pca.fit_transform(X)

    # return X_pca if not component else X_pca[component]
    return X_pca if not component else X_pca[:,:component]

def run_svm(layer, labels, num_trials, leave_out, cross_val=5):
    '''
    Runs SVM on the given layer over the number of trials and returns
    the accuracy of each trial

    Parameters:
        - layer: numpy array of NN layer
        - labels: numpy array of labels associated with each image (index)
        - num_trials: int representing the number of trials to run on layer
        - leave_out: bool representing whether or not to use leave_n_out
        - cross_val: int representing the number of cross validation folds

    Returns:
        - accuracy: list of accuracy (float) for each trial (length=num_trials)
    '''    
    accuracy = []

    for i in range(num_trials):
        if leave_out:
            train_layer, test_layer, y_train, y_test = leave_n_out(layer, labels, 2, 4)
        else:
            train_layer, test_layer, y_train, y_test = train_test_split(layer, labels, test_size=0.2)

        X_train = train_layer.reshape((train_layer.shape[0], -1))
        X_test = test_layer.reshape((test_layer.shape[0], -1))
        y_train, y_test = np.array(y_train), np.array(y_test)

        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        # scores = cross_val_score(clf, X_imgs, y_labs, cv=cross_val)
        accuracy.append(clf.score(X_test, y_test))

    return accuracy


def main(directory, img_paths, vgg_layers_path=None, num_trials=1, 
    use_pca=False, component_used=None, leave_out=False, cross_val=5):
    '''
    Parameters:
        - directory: str representing the path to the directory where the images are
        - img_paths: str represnting the txt file where each line has the file name of each image
        - vgg_layers_path: str representing the path to the directory 
            where the layers are (or should be saved)
        - num_trials: int representing the number of trials to run on layer
        - use_pca: bool representing whether or not to use PCA
        - component_used: int representing the component to use for PCA
        - leave_out: bool representing whether or not to use leave_n_out
        - cross_val: int representing the number of cross validation folds

    Returns:
        None
    '''
    output_file, plot_title, fig_name = build_names(num_trials, use_pca, component_used, leave_out)

    batch, batch_size, labels = loadImages(directory, img_paths)

    layers = getLayers(batch, batch_size, vgg_layers_path)


    # variance_explained = {}

    # TODO: PCA here save?
    
    results = {}
    for layer_name, layer in layers.items():

        if use_pca:
            
            layer = pca(batch_size, layer, component_used)

            # variance_explained[layer_name] = np.asarray(layer, dtype=float)

            

        accuracy = run_svm(layer, labels, num_trials, leave_out)
        results[layer_name] = accuracy


    # write_accuracies(output_file, variance_explained)

    write_accuracies(output_file, results)

    make_error_bar_plot(results, plot_title, fig_name)


if __name__ == '__main__':
    # MAKE SURE ALL PATHS ARE CORRECT
    directory = '/om/user/eeastman/interaction_images/'
    img_paths = directory + 'interaction_imgs.txt' # TEXT FILE WITH LIST OF IMAGE NAMES AND LABELS
                                                        # WHERE EACH LINE HAS "name_of_image.ext label"
    vgg_layers_path = '/om/user/eeastman/layers/vgg/'
    
    # PARAMETERS
    num_trials = 20
    use_pca = False
    pca_component_used = 49
    leave_out = False

    main(directory, img_paths, vgg_layers_path, num_trials, use_pca, pca_component_used, leave_out)
