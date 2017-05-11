# Module imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
####

# Global vars
matfile = loadmat('face_images.mat')
train_imgs = matfile['train_imgs']
test_imgs = matfile['test_imgs']
train_ids = matfile['train_ids']
test_ids = matfile['test_ids']

# Functions
def reshape_image_matrix(train_imgs):
    input_temp = []
    for i in range(len(train_imgs)):
        input_temp.append(train_imgs[i].reshape(1, -1)[0])
    return np.array(input_temp)
def generate_b(image_set):
    b = np.zeros(200*180)
    for i in range(len(image_set)):
        b += image_set[i]
    b /= len(image_set)
    return b
def pca(image_set):
    b = generate_b(image_set)
    X = image_set - b
    eigvecs, eigs, var = np.linalg.svd(X.T, full_matrices=False)
    idx = eigs.argsort()[::-1]
    eigs = eigs[idx]
    eigvecs = eigvecs[:, idx]
    sorted_eigs = sorted(eigs, reverse=True)
    return b, eigvecs, sorted_eigs
def fraction_of_variances(eigs):
    eigs = np.power(eigs, 2)
    sum_eigs = np.sum(eigs)
    cum_sum = np.cumsum(eigs)/sum_eigs
    dim_subspace = np.where(cum_sum >= 0.9)[0][0]
    return cum_sum, dim_subspace
def reconstruct_from_basis(W, b, image_set):
    x = np.dot(W.T, (image_set - b).T)
    y_recon = b + np.dot(x.T, W.T)
    error = np.power(abs(y_recon - image_set), 2)
    sample = np.random.permutation(range(len(image_set)))
    for i in range(5):
        index = sample[i]
        plt.suptitle("Square Reconstruction Error: {}".format(np.sum(error[index])))
        plt.subplot(131)
        plt.imshow(image_set[index].reshape((200, 180)))
        plt.subplot(132)
        plt.imshow(y_recon[index].reshape((200, 180)))
        plt.subplot(133)
        abs_dif = abs(y_recon[index] - image_set[index])
        plt.imshow(abs_dif.reshape((200, 180)))
        plt.show()
    return y_recon, x, error
def show_failed_classifers(error, false_indices, image_set, y_recon):
    for i in range(0, 5):
        index = false_indices[i]
        plt.suptitle("Square Reconstruction Error: {}".format(error[index]))
        plt.subplot(131)
        plt.imshow(image_set[index].reshape((200, 180)))
        plt.subplot(132)
        plt.imshow(y_recon[index].reshape((200, 180)))
        plt.subplot(133)
        abs_dif = abs(y_recon[index] - image_set[index])
        plt.imshow(abs_dif.reshape((200, 180)))
        plt.show()
def create_scatter_plot(image_set, id_set, dim1, dim2, subject_id):
    plt.title("Subject {}, Subspace Dimensions ({}, {})".format(subject_id, dim1 + 1, dim2 + 1))
    for i in range(len(image_set[0])):
        x = image_set[dim1][i]
        y = image_set[dim2][i]
        id = id_set[i][0]
        if id == subject_id:
            plt.scatter(x, y, color='brown')
        else:
            plt.scatter(x, y, color='yellow')
    plt.show()
def k_nn(train_imgs, train_subspace, train_ids, train_recon, test_imgs, test_subspace, test_ids, test_recon):
    accuracy = 0.0
    for i in range(len(test_imgs)):
        minVal = 100000000000000000
        best_match = 1000000000
        for j in range(len(train_imgs)):

            distance = np.sum(np.power(test_subspace[:, i] - train_subspace[:, j], 2))
            if distance < minVal:
                minVal = distance
                best_match = j
        if test_ids[i][0] == train_ids[best_match][0]:
            accuracy += 1.0
        else:
            plt.subplot(141)
            plt.imshow(test_imgs[i].reshape((200, 180)))
            plt.subplot(142)
            plt.imshow(test_recon[i].reshape((200, 180)))
            plt.subplot(143)
            plt.imshow(train_imgs[best_match].reshape((200, 180)))
            plt.subplot(144)
            plt.imshow(train_recon[best_match].reshape((200, 180)))
            plt.show()
    prob = accuracy/len(test_imgs)
    print "Accuracy of 1-NN is {}".format(prob)

# Step 1
for i in range(0, 3):
    plt.imshow(train_imgs[i])
    plt.show()

# Step 2
# Print Eigen value graphs
input_train = reshape_image_matrix(train_imgs)
b, eigvecs, eigs = pca(input_train)
plt.imshow(b.reshape((200, 180)))
plt.show()
plt.title("Scree Plot")
plt.plot(range(len(eigs)), eigs)
plt.show()

# Print fraction of variances graph
frac_of_vars, dimensionality = fraction_of_variances(eigs)
print dimensionality
plt.title("Fraction of Variance Explained")
plt.plot(range(len(frac_of_vars)), frac_of_vars)
plt.show()
for i in range(10):
    num = i+1
    plt.title("Eigenvector {}".format(num))
    plt.imshow(eigvecs[:, i].reshape((200, 180)))
    plt.show()


# STEP 3

# Weight parameter
W = eigvecs[:, 0: dimensionality]

y_face_recon, face_subspace, face_error = reconstruct_from_basis(W, b, input_train)

face_error_sums = np.array([np.sum(face_error[i]) for i in range(len(face_error))])

# Load non face images
matfile_non = loadmat('nonface_images.mat')
train_imgs_non = matfile_non['nonface_imgs']
non_input_train = reshape_image_matrix(train_imgs_non)
y_non_recon, non_subspace, non_error = reconstruct_from_basis(W, b, non_input_train)
non_error_sums = np.array([sum(non_error[i]) for i in range(len(non_error))])
# Show errors in histogram
plt.hist(non_error_sums, bins='auto',  label='x')
plt.hist(face_error_sums, bins='auto',  label='y')
plt.legend(loc='upper right')
plt.show()
# set threshold
threshold = 7375
# Faces classified as non faces
false_non_faces = np.where(face_error_sums > threshold)[0]
# Non faces classified as faces
false_faces = np.where(non_error_sums < threshold)[0]
show_failed_classifers(face_error_sums, false_non_faces, input_train, y_face_recon)
show_failed_classifers(non_error_sums, false_faces, non_input_train, y_non_recon)
# STEP 4

# Create scatter plot for subject 1
create_scatter_plot(face_subspace, train_ids, 0, 1, 1)
create_scatter_plot(face_subspace, train_ids, 1, 2, 1)
create_scatter_plot(face_subspace, train_ids, 2, 3, 1)
# Create scatter plot for subject 2
create_scatter_plot(face_subspace, train_ids, 0, 1, 2)
create_scatter_plot(face_subspace, train_ids, 1, 2, 2)
create_scatter_plot(face_subspace, train_ids, 2, 3, 2)

W = eigvecs[:, 0: 10]
test_input = reshape_image_matrix(test_imgs)
y_test_recon, test_subspace, test_error = reconstruct_from_basis(W, b, test_input)
k_nn(input_train, face_subspace[:10], train_ids, y_face_recon, test_input, test_subspace, test_ids, y_test_recon)


