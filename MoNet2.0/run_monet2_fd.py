import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model, Model
from utils_monet2 import set_random_seed, generate_classification, generate_brownian, generate_fbm, generate_ctrw, generate_lw, generate_sbm, generate_attm, normalize_trajectory
from scipy.linalg import sqrtm

# Set environment variables and random seed
os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_random_seed(4466)

# Load the pre-trained classification model
model = load_model('monet2.keras')

# Load and normalize LPTEM trajectories
traj_lptem = np.load('lptem_val.npy')[:, :, 0:2]
normalized_lptem = []
for i in range(traj_lptem.shape[0]):
    normalized_lptem.append(np.asarray(normalize_trajectory(traj_lptem[i,:,0], traj_lptem[i,:,1])))

normalized_lptem = np.transpose(np.asarray(normalized_lptem),(0,2,1))
np.random.shuffle(normalized_lptem)

# Load and normalize generated trajectories
generated_trajectories = np.load('LEONARDO_generated.npy')
normalized_generated_trajectories = []
for i in range(generated_trajectories.shape[0]):
    normalized_generated_trajectories.append(np.asarray(normalize_trajectory(generated_trajectories[i, :, 0], generated_trajectories[i, :, 1])))
normalized_generated_trajectories = np.transpose(np.asarray(normalized_generated_trajectories), (0, 2, 1))

# Generate test data for classification
batchsize = int(3000 * 6)
steps = 200
dim = 2
x_test, y_test = generate_classification(batchsize=batchsize, steps=steps, dim=dim, include_lptem=traj_lptem[0:3000])

# Create a feature extractor model from the designated layer
feature_layer = 'dense_1'
monet_feature_extractor = Model(inputs=model.input, outputs=model.get_layer(feature_layer).output)
real_features = monet_feature_extractor.predict(x_test)

def extract_features(feature_extractor, trajectories):
    return feature_extractor.predict(trajectories)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    sigma1 = sigma1 + np.eye(sigma1.shape[0])
    sigma2 = sigma2 + np.eye(sigma2.shape[0])
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    fid = np.sqrt((np.sum(diff**2)) + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid

def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_fid_matrix(feature_dict):
    models = list(feature_dict.keys())
    fid_matrix = np.zeros((len(models), len(models)))
    for i, model_1 in enumerate(models):
        mu1, sigma1 = compute_statistics(feature_dict[model_1])
        for j, model_2 in enumerate(models):
            mu2, sigma2 = compute_statistics(feature_dict[model_2])
            fid_matrix[i, j] = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_matrix, models

# ---------------------- First FID Matrix ----------------------
# Split generated trajectories into two batches
num_generated = normalized_generated_trajectories.shape[0]
batchsize_gen = num_generated // 2
generated_trajectories_1 = normalized_generated_trajectories[:batchsize_gen]
generated_trajectories_2 = normalized_generated_trajectories[batchsize_gen:]

# Generate trajectories for each diffusion model (first set)
batchsize_model = int(normalized_lptem.shape[0] / 2)
brownian_trajectories_1 = generate_brownian(batchsize_model, steps, dim)
brownian_trajectories_2 = generate_brownian(batchsize_model, steps, dim)
fbm_trajectories_1 = generate_fbm(batchsize_model, steps, dim)
fbm_trajectories_2 = generate_fbm(batchsize_model, steps, dim)
ctrw_trajectories_1 = generate_ctrw(batchsize_model, steps, dim)
ctrw_trajectories_2 = generate_ctrw(batchsize_model, steps, dim)
lw_trajectories_1 = generate_lw(batchsize_model, steps, dim)
lw_trajectories_2 = generate_lw(batchsize_model, steps, dim)
sbm_trajectories_1 = generate_sbm(batchsize_model, steps, dim)
sbm_trajectories_2 = generate_sbm(batchsize_model, steps, dim)
attm_trajectories_1 = generate_attm(batchsize_model, steps, dim)
attm_trajectories_2 = generate_attm(batchsize_model, steps, dim)

# Extract features for each diffusion model (first set)
feature_dict_full = {}
feature_dict_full['Brownian 1'] = extract_features(monet_feature_extractor, brownian_trajectories_1)
feature_dict_full['Brownian 2'] = extract_features(monet_feature_extractor, brownian_trajectories_2)
feature_dict_full['FBM 1'] = extract_features(monet_feature_extractor, fbm_trajectories_1)
feature_dict_full['FBM 2'] = extract_features(monet_feature_extractor, fbm_trajectories_2)
feature_dict_full['CTRW 1'] = extract_features(monet_feature_extractor, ctrw_trajectories_1)
feature_dict_full['CTRW 2'] = extract_features(monet_feature_extractor, ctrw_trajectories_2)
feature_dict_full['LW 1'] = extract_features(monet_feature_extractor, lw_trajectories_1)
feature_dict_full['LW 2'] = extract_features(monet_feature_extractor, lw_trajectories_2)
feature_dict_full['SBM 1'] = extract_features(monet_feature_extractor, sbm_trajectories_1)
feature_dict_full['SBM 2'] = extract_features(monet_feature_extractor, sbm_trajectories_2)
feature_dict_full['ATTM 1'] = extract_features(monet_feature_extractor, attm_trajectories_1)
feature_dict_full['ATTM 2'] = extract_features(monet_feature_extractor, attm_trajectories_2)
feature_dict_full['LPTEM 1'] = extract_features(monet_feature_extractor, normalized_lptem[:batchsize_model])
feature_dict_full['LPTEM 2'] = extract_features(monet_feature_extractor, normalized_lptem[batchsize_model:])
feature_dict_full['Generated 1'] = extract_features(monet_feature_extractor, generated_trajectories_1)
feature_dict_full['Generated 2'] = extract_features(monet_feature_extractor, generated_trajectories_2)

fid_matrix_full, models_full = calculate_fid_matrix(feature_dict_full)

mask_full = np.triu(np.ones_like(fid_matrix_full, dtype=bool))
fig1 = plt.figure(figsize=(15, 8))
sns.heatmap(fid_matrix_full, xticklabels=models_full, yticklabels=models_full, annot=True,
            cmap='rocket_r', fmt='.2f', mask=mask_full)
plt.title('FID Matrix Between Diffusion Classes (Lower Triangular)')
plt.tight_layout()
plt.savefig("fid_matrix_separate_batches.png", format="png", dpi=300)
plt.close(fig1)

# ---------------------- Second FID Matrix ----------------------
# For the second matrix, use all available trajectories
batchsize_model = normalized_lptem.shape[0]
brownian_trajectories = generate_brownian(batchsize_model, steps, dim)
fbm_trajectories = generate_fbm(batchsize_model, steps, dim)
ctrw_trajectories = generate_ctrw(batchsize_model, steps, dim)
lw_trajectories = generate_lw(batchsize_model, steps, dim)
sbm_trajectories = generate_sbm(batchsize_model, steps, dim)
attm_trajectories = generate_attm(batchsize_model, steps, dim)

# Extract features for each diffusion model (second set)
feature_dict = {}
feature_dict['Brownian'] = extract_features(monet_feature_extractor, brownian_trajectories)
feature_dict['FBM'] = extract_features(monet_feature_extractor, fbm_trajectories)
feature_dict['CTRW'] = extract_features(monet_feature_extractor, ctrw_trajectories)
feature_dict['LW'] = extract_features(monet_feature_extractor, lw_trajectories)
feature_dict['SBM'] = extract_features(monet_feature_extractor, sbm_trajectories)
feature_dict['ATTM'] = extract_features(monet_feature_extractor, attm_trajectories)
feature_dict['LPTEM'] = extract_features(monet_feature_extractor, normalized_lptem)
feature_dict['Generated'] = extract_features(monet_feature_extractor, normalized_generated_trajectories)

fid_matrix, models_second = calculate_fid_matrix(feature_dict)

mask = np.triu(np.ones_like(fid_matrix, dtype=bool))
fig2 = plt.figure(figsize=(9, 4.8))
sns.heatmap(fid_matrix, xticklabels=models_second, yticklabels=models_second, annot=True,
            cmap='rocket_r', fmt='.2f', mask=mask)
plt.tight_layout()
plt.savefig("fid_matrix.png", format="png", dpi=300)
plt.close(fig2)
