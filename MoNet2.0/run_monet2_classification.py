import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils_monet2 import generate_classification, normalize_trajectory, set_random_seed

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
    normalized_generated_trajectories.append(np.asarray(normalize_trajectory(generated_trajectories[i,:,0], generated_trajectories[i,:,1])))

normalized_generated_trajectories = np.transpose(np.asarray(normalized_generated_trajectories),(0,2,1))


# Generate test data for classification
batchsize = int(3000 * 6)
steps = 200
dim = 2
x_test, y_test = generate_classification(batchsize=batchsize, steps=steps, dim=dim, include_lptem=traj_lptem[0:3000])

# Predict on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Compute and normalize the confusion matrix to percentages
disp_labels = ['BM', 'FBM', 'CTRW', 'LW', 'SBM', 'ATTM', 'LPTEM']
cm = confusion_matrix(y_true_classes, y_pred_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix with a reversed rocket colormap
rocket_r = sns.color_palette("rocket", as_cmap=True).reversed()
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=disp_labels)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap=rocket_r, ax=ax, values_format='.2f', colorbar=False)

# Add a horizontal colorbar at the bottom
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap=rocket_r, norm=plt.Normalize(vmin=0, vmax=100)),
    ax=ax,
    orientation='horizontal',
    fraction=0.046,
    pad=0.1
)
cbar.ax.tick_params(labelsize=15)
cbar.set_label('Percentage (%)', fontsize=18)

# Adjust x-axis labels and font sizes
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.xaxis.labelpad = 5
plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
for text in disp.text_.ravel():
    text.set_fontsize(15)
ax.set_xlabel('Predicted Label', fontsize=18)
ax.set_ylabel('True Label', fontsize=18)
plt.tight_layout()

# Save the confusion matrix figure and close it
plt.savefig("confusion_matrix.png", format="png", dpi=300)
plt.close(fig)

# Predict on the generated trajectories
generated_y_pred = model.predict(normalized_generated_trajectories)
generated_pred_classes = np.argmax(generated_y_pred, axis=1)
class_counts = np.bincount(generated_pred_classes, minlength=len(disp_labels))
total_generated = len(normalized_generated_trajectories)
class_percentages = (class_counts / total_generated) * 100

# Plot bar chart for the classification of generated trajectories
plt.figure(figsize=(10, 6))
bars = plt.bar(disp_labels, class_counts)
for bar, percentage in zip(bars, class_percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
        f"{percentage:.2f}%",
        ha='center', va='bottom', fontsize=10
    )
plt.title('Classification of Generated Trajectories')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the bar chart figure and close it
plt.savefig("generated_trajectories_classification.png", format="png", dpi=300)
plt.close()
