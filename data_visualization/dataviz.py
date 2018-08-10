# -*- coding: utf-8 -*-
import sys, os

sys.path.append('../')
from pytorch.common.datasets_parsers.av_parser import AVDBParser

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from t_sne.tsne import TSNE as torchTSNE
from scipy.spatial.distance import squareform
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

from data_visualization.t_sne.wrapper import Wrapper
from pytorch.common.datasets_parsers.av_parser import AVDBParser
from tqdm import tqdm


def get_data(dataset_root, file_list, max_num_clips=0):
    dataset_parser = AVDBParser(dataset_root, os.path.join(dataset_root, file_list),
                                max_num_clips=max_num_clips, ungroup=False, load_image=False)
    data = dataset_parser.get_data()
    print('clips count:', len(data))
    print('frames count:', dataset_parser.get_dataset_size())
    return data


def calc_features(data):
    orb = cv2.ORB_create()

    progresser = tqdm(iterable=range(0, len(data)),
                      desc='calc video features',
                      total=len(data),
                      unit='files')

    feat, targets = [], []
    for i in progresser:
        clip = data[i]

        if not clip.data_samples[0].labels in [7, 8]:
            continue

        # TODO: придумайте способы вычисления признаков по изображению с использованием ключевых точек
        # используйте библиотеку OpenCV
        if 1:  # distance between landmarks
            for i, sample in enumerate(clip.data_samples):
                if i % 8 != 0:
                    continue
                dist = []
                lm_ref = sample.landmarks[30]  # point on the nose
                for j in range(len(sample.landmarks)):
                    lm = sample.landmarks[j]
                    dist.append(np.sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))
                feat.append(dist)
                targets.append(sample.labels)
        elif 1:  # descriptors of landmarks
            rm_list = []
            for i, sample in enumerate(clip.data_samples):
                if i % 8 != 0:
                    continue
                # make image border
                return None
                bordersize = 15
                border = cv2.copyMakeBorder(sample.image, top=bordersize, bottom=bordersize, left=bordersize,
                                            right=bordersize,
                                            borderType=cv2.BORDER_CONSTANT, value=[0] * 3)

                # make keypoint list
                keypoints = []
                for k in range(18, 68):
                    keypoints.append(cv2.KeyPoint(x=sample.landmarks[k][0] + bordersize,
                                                  y=sample.landmarks[k][1] + bordersize,
                                                  _size=128))

                # compute the descriptors with ORB
                keypoints_actual, descriptors = orb.compute(border, keypoints)
                if len(keypoints_actual) != len(keypoints):
                    rm_list.append(sample)
                    continue

                descriptors = np.concatenate(descriptors)
                feat.append(descriptors)
                targets.append(sample.labels)

            for sample in rm_list:
                clip.data_samples.remove(sample)
        elif 1:
            for i, sample in enumerate(clip.data_samples):
                if i % 8 != 0:
                    continue
                dist = pairwise_distances(np.asarray(sample.landmarks), metric='euclidean', squared=True)
                feat.append(dist[np.triu_indices(dist.shape[0], 1)].tolist())
                targets.append(sample.labels)
        elif 1:
            for i, sample in enumerate(clip.data_samples):
                if i % 8 != 0:
                    continue
                dist = []
                lm = np.array(sample.landmarks)
                xmean, ymean = lm.mean(axis=0)
                distance = pairwise_distances(lm, np.array([[xmean, ymean]]), metric='euclidean', squared=True)
                lmcentered = lm - np.array([[xmean, ymean]])
                angle = [(math.atan2(lmc[1], lmc[0]) * 360) / (2 * math.pi) for lmc in lmcentered]
                alpha = (math.atan2(sample.landmarks[27][1] - sample.landmarks[30][1],
                                    sample.landmarks[27][0] - sample.landmarks[30][0]) * 360) / (2 * math.pi)
                phi = -90. - alpha
                angle = [ang + phi for ang in angle]
                dist = distance.ravel().tolist() + angle
                feat.append(dist)
                targets.append(sample.labels)
        elif 1:
            dist = []
            lm_ref = clip.data_samples[len(clip.data_samples) // 2].landmarks
            for i, sample in enumerate(clip.data_samples):
                lm = sample.landmarks
                bias = []
                for j in range(len(lm_ref)):
                    bias.append((abs(lm_ref[j][0] - lm[j][0]) + abs(lm_ref[j][1] - lm[j][1])) / 2)
                dist.append(bias)
            feat.append(np.std(dist, axis=0))
            targets.append(clip.data_samples[0].labels)
        else:
            for i, sample in enumerate(clip.data_samples):
                if i % 8 != 0:
                    continue
                dist = []
                dist.append(np.sqrt((sample.landmarks[22][0] - sample.landmarks[23][0]) ** 2
                                    + (sample.landmarks[22][1] - sample.landmarks[23][1]) ** 2))
                dist.append(np.abs(sample.landmarks[22][1] - sample.landmarks[28][1]))
                dist.append(np.sqrt((sample.landmarks[52][0] - sample.landmarks[58][0]) ** 2
                                    + (sample.landmarks[52][1] - sample.landmarks[58][1]) ** 2))
                dist.append(np.sqrt((sample.landmarks[49][0] - sample.landmarks[55][0]) ** 2
                                    + (sample.landmarks[49][1] - sample.landmarks[55][1]) ** 2))
                dist.append(np.abs(sample.landmarks[49][0] - sample.landmarks[52][0]))
                dist.append(np.abs(sample.landmarks[49][1] - sample.landmarks[52][1]))

                feat.append(dist)
                targets.append(sample.labels)

    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def draw(points2D, targets, sawe=False):
    fig = plt.figure()
    plt.scatter(points2D[:, 0], points2D[:, 1], c=targets)
    plt.axis('off')
    if sawe:
        plt.savefig('scatter.png', bbox_inches='tight')
        plt.close(fig)
    else:
        fig.show()
        plt.pause(5)
        plt.close(fig)


def run_tsne(feat, targets, pca_dim=50, tsne_dim=2):
    if pca_dim > 0:
        feat = PCA(n_components=pca_dim).fit_transform(feat)

    distances2 = pairwise_distances(feat, metric='euclidean', squared=True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances2, 30, False)
    # Convert to n x n prob array
    pij = squareform(pij)

    i, j = np.indices(pij.shape)
    i, j = i.ravel(), j.ravel()
    pij = pij.ravel().astype('float32')
    # Remove self-indices
    idx = i != j
    i, j, pij = i[idx], j[idx], pij[idx]

    model = torchTSNE(n_points=feat.shape[0], n_dim=tsne_dim)
    w = Wrapper(model, batchsize=feat.shape[0], epochs=5)
    for itr in range(5):
        w.fit(pij, i, j)
        # Visualize the results
        embed = model.logits.weight.cpu().data.numpy()
        draw(embed, targets)


if __name__ == "__main__":
    # dataset dir
    base_dir = '/media/stc_ml_school/data'
    if 1:
        train_dataset_root = base_dir + '/Ryerson/Video'
        train_file_list = base_dir + '/Ryerson/train_data_with_landmarks.txt'
    elif 0:
        train_dataset_root = base_dir + '/AFEW-VA/crop'
        train_file_list = base_dir + '/AFEW-VA/crop/train_data_with_landmarks.txt'
    elif 1:
        train_dataset_root = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames'
        train_file_list = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt'

    # load dataset
    data = get_data(train_dataset_root, train_file_list, max_num_clips=0)

    # get features
    feat, targets = calc_features(data)

    # run t-SNE
    run_tsne(feat, targets, pca_dim=0)