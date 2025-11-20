import numpy as np
import cv2
import os
import pandas as pd
from matplotlib import pyplot as plt
import statistics
import time

DB_PATH = r'C:/Users/maria/Desktop/ORL'
IMAGES_PER_PERSON = 10
TRAINING_IMAGES = 8
NUM_PEOPLE = 40
PIXELS_COUNT = 10304

def nearest_neighbor(train_matrix, test_img, distance_metric):
    """Find nearest neighbor using specified distance metric."""
    distances = np.zeros(len(train_matrix[0]))

    for i in range(len(train_matrix[0])):
        if distance_metric == 'cos':
            distances[i] = 1 - np.dot(train_matrix[:, i], test_img) / \
                           (np.linalg.norm(train_matrix[:, i]) * np.linalg.norm(test_img))
        elif distance_metric == 'inf':
            distances[i] = np.linalg.norm(train_matrix[:, i] - test_img, np.inf)
        else:
            distances[i] = np.linalg.norm(train_matrix[:, i] - test_img, float(distance_metric))

    return np.argmin(distances)

def k_nearest_neighbors(train_matrix, test_img, distance_metric, k):
    """Find k nearest neighbors and return mode of their labels."""
    distances = np.zeros(len(train_matrix[0]))

    for i in range(len(train_matrix[0])):
        if distance_metric == 'cos':
            distances[i] = 1 - np.dot(train_matrix[:, i], test_img) / \
                           (np.linalg.norm(train_matrix[:, i]) * np.linalg.norm(test_img))
        elif distance_metric == 'inf':
            distances[i] = np.linalg.norm(train_matrix[:, i] - test_img, np.inf)
        else:
            distances[i] = np.linalg.norm(train_matrix[:, i] - test_img, float(distance_metric))

    nearest_indices = np.argsort(distances)[:k]
    person_ids = nearest_indices // TRAINING_IMAGES + 1
    most_common_person = statistics.mode(person_ids)

    return (most_common_person - 1) * TRAINING_IMAGES

def build_training_matrix(database_path):
    """Build training matrix from database images."""
    matrix = np.zeros((PIXELS_COUNT, TRAINING_IMAGES * NUM_PEOPLE))

    for person_id in range(1, NUM_PEOPLE + 1):
        for img_id in range(1, TRAINING_IMAGES + 1):
            img_path = os.path.join(database_path, f's{person_id}', f'{img_id}.pgm')
            image = cv2.imread(img_path, 0)
            matrix[:, (person_id - 1) * TRAINING_IMAGES + (img_id - 1)] = np.reshape(image, (PIXELS_COUNT,))

    return matrix

def evaluate_classifier(train_matrix, classifier_func, metrics_list, k_value=None):
    """Evaluate classifier performance across different distance metrics."""
    results_list = []
    test_images_count = IMAGES_PER_PERSON - TRAINING_IMAGES
    total_evaluations = NUM_PEOPLE * test_images_count

    for metric in metrics_list:
        correct_predictions = 0
        elapsed_time = 0

        for person_id in range(1, NUM_PEOPLE + 1):
            for img_id in range(TRAINING_IMAGES + 1, IMAGES_PER_PERSON + 1):
                test_img_path = os.path.join(DB_PATH, f's{person_id}', f'{img_id}.pgm')
                test_image = np.reshape(cv2.imread(test_img_path, 0), (-1,))

                time_start = time.time()
                if k_value is None:
                    prediction_idx = classifier_func(train_matrix, test_image, metric)
                else:
                    prediction_idx = classifier_func(train_matrix, test_image, metric, k_value)
                elapsed_time += time.time() - time_start

                predicted_person = prediction_idx // TRAINING_IMAGES + 1

                if predicted_person == person_id:
                    correct_predictions += 1

        recognition_rate = correct_predictions / total_evaluations
        avg_query_time = elapsed_time / total_evaluations

        results_list.append({
            'Algorithm': 'NN' if k_value is None else 'KNN',
            'K': np.nan if k_value is None else k_value,
            'Distance_Metric': metric,
            'Recognition_Rate': recognition_rate,
            'Avg_Query_Time': avg_query_time
        })

        algo_name = 'NN' if k_value is None else f'{k_value}-NN'
        print(f"{algo_name}(norm = {metric}): RR={recognition_rate * 100:.2f}%  AQT (s)={avg_query_time:.5f}")
    return results_list

def predict_demo_image(train_matrix, classifier_func, metrics_list, demo_image, k_value=None):
    """Generate predictions for a demo image using all distance metrics."""
    predictions_dict = {}

    for metric in metrics_list:
        if k_value is None:
            pred_idx = classifier_func(train_matrix, demo_image, metric)
        else:
            pred_idx = classifier_func(train_matrix, demo_image, metric, k_value)

        predicted_person = pred_idx // TRAINING_IMAGES + 1
        predictions_dict[metric] = (pred_idx, predicted_person)

    return predictions_dict

def visualize_predictions(train_matrix, demo_image, prediction_results):
    """Display test image alongside predicted matches for each distance metric."""
    num_subplots = len(prediction_results) + 1
    plt.figure(figsize=(3 * num_subplots, 4))

    plt.subplot(1, num_subplots, 1)
    plt.imshow(np.reshape(demo_image, (112, 92)), cmap='gray')
    plt.title("Test Image")
    plt.axis('off')

    for subplot_idx, (metric, (matrix_idx, person)) in enumerate(prediction_results.items()):
        plt.subplot(1, num_subplots, subplot_idx + 2)
        plt.imshow(np.reshape(train_matrix[:, matrix_idx], (112, 92)), cmap='gray')
        plt.title(f"NN - {metric}\nP.{person}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_performance_from_csv(csv_filepath, algorithm_type, k_parameter=None):
    """Generate performance plots from CSV data."""
    dataframe = pd.read_csv(csv_filepath)

    if k_parameter is None:
        filtered_data = dataframe[dataframe['Algorithm'] == algorithm_type]
        plot_title = "NN"
    else:
        filtered_data = dataframe[(dataframe['Algorithm'] == algorithm_type) &
                                  (dataframe['K'] == k_parameter)]
        plot_title = f"KNN (k={k_parameter})"

    metric_names = filtered_data['Distance_Metric'].values
    recognition_rates = filtered_data['Recognition_Rate'].values
    query_times = filtered_data['Avg_Query_Time'].values

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(metric_names, recognition_rates, marker='o', linestyle='-', color='blue')
    plt.title(f'{plot_title} - Recognition Rate')
    plt.ylabel("Recognition Rate")
    plt.xlabel("Distance Metric")
    plt.ylim(0, 1)

    for i, rate in enumerate(recognition_rates):
        plt.text(metric_names[i], rate + 0.02, f'{rate:.4f}', ha='center', fontsize=8)

    plt.subplot(1, 2, 2)
    plt.plot(metric_names, query_times, marker='o', linestyle='-', color='red')
    plt.title(f'{plot_title} - Average Query Time (s)')
    plt.ylabel("Time (s)")
    plt.xlabel("Distance Metric")

    for i, time_val in enumerate(query_times):
        plt.text(metric_names[i], time_val + time_val * 0.05, f'{time_val:.5f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()

training_matrix = build_training_matrix(DB_PATH)

distance_metrics = ['1', '2', 'inf', 'cos']
k_neighbors_list = [3, 5, 7, 9, 11]

demo_test_image = np.reshape(cv2.imread(os.path.join(DB_PATH, 's1', '9.pgm'), 0), (-1,))

all_results = []

print("\nEVALUATING NEAREST NEIGHBOR (NN)")
nn_results = evaluate_classifier(training_matrix, nearest_neighbor, distance_metrics)
all_results.extend(nn_results)

print("\nEVALUATING K-NEAREST NEIGHBORS (KNN)")
for k_val in k_neighbors_list:
    print(f"\n--- Testing with k={k_val} ---")
    knn_results = evaluate_classifier(training_matrix, k_nearest_neighbors, distance_metrics, k_value=k_val)
    all_results.extend(knn_results)

results_dataframe = pd.DataFrame(all_results, columns=['Algorithm', 'K', 'Distance_Metric',
                                                       'Recognition_Rate', 'Avg_Query_Time'])
results_dataframe.to_csv('statistics.csv', index=False)
print("\nResults saved to 'statistics.csv'")

nn_demo_predictions = predict_demo_image(training_matrix, nearest_neighbor, distance_metrics, demo_test_image)
visualize_predictions(training_matrix, demo_test_image, nn_demo_predictions)

print("\nGENERATING PERFORMANCE PLOTS")
plot_performance_from_csv('statistics.csv', 'NN')

for k_val in k_neighbors_list:
    plot_performance_from_csv('statistics.csv', 'KNN', k_parameter=k_val)

print("\nAll plots generated successfully!")