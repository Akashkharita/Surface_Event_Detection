#!/home/ahutko/miniconda3/envs/surface_dl/bin/python

"""
Seismic Event Classification Analysis with Magnitude Distribution

This script analyzes seismic event classification results from multiple models,
evaluates their performance, and analyzes the relationship between event magnitude
and prediction accuracy.

Features:
- Identifies the best parameter configurations for each model and event type
- Calculates performance metrics (precision, recall, F1, accuracy)
- Creates magnitude-based histograms for correct vs. incorrect predictions
- Generates confusion matrices and performance metric visualizations
- Writes detailed performance results to output files
- Handles empty files and detects/stops at repeated lines

Usage:
  python seismic_analysis.py

Inputs:
  - RESULTS/*ou*.txt files: Model prediction results
  - earthquake_data.txt: Contains event magnitudes and analyst classifications

Outputs:
  - Text reports of model performance
  - Visualizations of model performance metrics
  - Magnitude histograms for correct/incorrect predictions by model
"""

import glob
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------------------------------------------------------
# Configuration Variables
# -----------------------------------------------------------------------------
RESULTS_PATTERN = 'RESULTS3/6199*ou*.txt'  # Pattern for model output files
OUTPUT_DIR = 'output'  # Directory for output files
BIN_WIDTH = 0.2  # Width of histogram bins
COLORS = {'eq': 'blue', 'ex': 'red', 'su': 'green'}  # Colors for each event type
OUTPUT_DPI = 300  # Resolution of output images
MIN_MAGNITUDE = -2.0  # Cap for extremely negative magnitudes

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def init_nested_dict():
    """Helper function to create nested defaultdicts."""
    return defaultdict(int)

def ensure_output_dir():
    """Ensure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
# Model evaluation data structures
correct_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # model -> params -> event_type -> count
total_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))    # model -> params -> event_type -> count
overall_correct = defaultdict(lambda: defaultdict(int))                           # model -> params -> count
overall_total = defaultdict(lambda: defaultdict(int))                             # model -> params -> count
paramcount = defaultdict(lambda: defaultdict(int))                                # model -> params -> count
confusion_matrices = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))  # model -> params -> true -> pred -> count
seismogram_counts = defaultdict(lambda: defaultdict(list))                        # model -> params -> [counts]

# Magnitude data structures
model_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> correct/incorrect -> event_type -> [magnitudes]

# List of event types for iteration
event_types = ['eq', 'ex', 'su']

# -----------------------------------------------------------------------------
# Data Loading Functions
# -----------------------------------------------------------------------------

def process_model_outputs():
    """
    Process all model output files.
    
    This function processes all output files, but doesn't build the histogram data yet.
    That will be done after we identify the single best parameter set for each model.
    """
    print(f"Processing model output files from {RESULTS_PATTERN}...")
    
    file_count = 0
    empty_count = 0
    repetition_count = 0
    
    # Save raw events data for later histogram creation
    raw_event_data = {}  # {evid: {"model": model, "params": params, "pred": pred, "analyst": analyst, "magnitude": mag}}
    
    for outfile in glob.glob(RESULTS_PATTERN):
        file_count += 1
        if file_count % 100 == 0:
            print(f"Processing file {file_count}...")
        
        # Read all lines from the file
        with open(outfile, 'r') as f:
            lines = f.readlines()
        
        # Skip empty files
        if not lines:
            empty_count += 1
            continue
        
        # Track processed lines to detect repetitions
        processed_lines = set()
        
        for line in lines:
            # Skip lines that don't contain "Analyst:"
            if "Analyst:" not in line:
                continue
                
            # Check for repeated lines
            if line in processed_lines:
                repetition_count += 1
                break  # Stop processing this file when repetition detected
            
            # Add line to processed set
            processed_lines.add(line)
            
            try:
                parts = line.split()
                if len(parts) < 21:
                    continue
                    
                evid = int(parts[0])
                name = parts[1]
                model = name.split('_')[0]  # e.g. SeismicCNN1d 

                # Extracting params using regex to split on either 'c_' or 'd_'
                params = re.split(r'c_|d_', name)[1] if len(re.split(r'c_|d_', name)) > 1 else ""
                
                # Extract the predicted class (index 14) and convert to lowercase
                pred = parts[14].lower()
                
                # Extract seismogram count (index 16)
                seismogramcount = int(parts[16])
                
                # Extract analyst label (index 18) and convert px to ex
                analyst = parts[18].lower()
                if analyst == 'px':
                    analyst = 'ex'

                # Extract magnitude
                magstr = parts[20]
                magnitude = float(magstr[2:])

                # Store seismogram count
                seismogram_counts[model][params].append(seismogramcount)
                
                # Update paramcount (total evaluations for this model+params)
                paramcount[model][params] += 1
                
                # Save the raw event data for later histogram creation
                if evid not in raw_event_data:
                    raw_event_data[evid] = []
                raw_event_data[evid].append({
                    "model": model,
                    "params": params,
                    "pred": pred,
                    "analyst": analyst,
                    "magnitude": magnitude
                })
                
                # Skip if prediction is "no" (noise) as it's always wrong per instructions
                if pred == "no":
                    # Still count it as a prediction for the true class
                    total_predictions[model][params][analyst] += 1
                    overall_total[model][params] += 1
                    # Update confusion matrix
                    confusion_matrices[model][params][analyst][pred] += 1
                    continue
                
                # Update prediction counts
                total_predictions[model][params][pred] += 1
                overall_total[model][params] += 1
                
                # Update correct prediction counts if prediction matches analyst
                if pred == analyst:
                    correct_predictions[model][params][pred] += 1
                    overall_correct[model][params] += 1
                
                # Update confusion matrix
                confusion_matrices[model][params][analyst][pred] += 1
                
            except (IndexError, ValueError) as e:
                # Skip lines that don't match expected format
                continue

    print(f"Processed {file_count} files.")
    print(f"  - Skipped {empty_count} empty files")
    print(f"  - Detected and handled {repetition_count} files with repeated lines")
    
    return raw_event_data

# -----------------------------------------------------------------------------
# Analysis Functions
# -----------------------------------------------------------------------------
def find_best_parameters():
    """
    Find the best parameters for each model and event type.
    
    Returns:
    - Dictionary of best parameters by model and event type
    - Dictionary of best overall parameters by model
    """
    # Find best params for each model and event type
    best_params_by_event = {}
    for model in correct_predictions:
        best_params_by_event[model] = {}
        
        for event_type in event_types:
            best_params = []
            best_accuracy = 0.0
            
            for params in total_predictions[model]:
                if total_predictions[model][params][event_type] > 0:
                    accuracy = correct_predictions[model][params][event_type] / total_predictions[model][params][event_type]
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = [params]
                    elif accuracy == best_accuracy:
                        best_params.append(params)
            
            best_params_by_event[model][event_type] = best_params

    # Find best overall params for each model
    best_overall_params = {}
    for model in overall_total:
        best_params = []
        best_accuracy = 0.0
        
        for params in overall_total[model]:
            if overall_total[model][params] > 0:
                accuracy = overall_correct[model][params] / overall_total[model][params]
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = [params]
                elif accuracy == best_accuracy:
                    best_params.append(params)
        
        # Break ties using median seismogram count
        if len(best_params) > 1:
            best_median = 0
            chosen_param = best_params[0]
            
            for param in best_params:
                if len(seismogram_counts[model][param]) > 0:
                    median_count = np.median(seismogram_counts[model][param])
                    if median_count > best_median:
                        best_median = median_count
                        chosen_param = param
            
            best_overall_params[model] = chosen_param
        else:
            best_overall_params[model] = best_params[0] if best_params else None

    return best_params_by_event, best_overall_params

def calculate_metrics(best_overall_params):
    """
    Calculate performance metrics for each model's best parameters.
    
    Returns dictionary of metrics by model.
    """
    metrics = {}
    for model in best_overall_params:
        if best_overall_params[model]:
            params = best_overall_params[model]
            metrics[model] = {
                'params': params,
                'correct': overall_correct[model][params],
                'total': overall_total[model][params],
                'accuracy': overall_correct[model][params] / overall_total[model][params] if overall_total[model][params] > 0 else 0,
                'confusion_matrix': confusion_matrices[model][params],
                'mean_seismogram_count': np.mean(seismogram_counts[model][params]) if seismogram_counts[model][params] else 0,
                'median_seismogram_count': np.median(seismogram_counts[model][params]) if seismogram_counts[model][params] else 0
            }

            # Calculate precision, recall, F1 for each class
            precision = {}
            recall = {}
            f1 = {}
            
            cm = confusion_matrices[model][params]
            for event_type in event_types:
                # True positives: Predicted this class correctly
                tp = cm[event_type][event_type] if event_type in cm and event_type in cm[event_type] else 0
                
                # False positives: Predicted this class but was wrong
                fp = sum(cm[true][event_type] if event_type in cm[true] else 0 
                         for true in cm if true != event_type)
                
                # False negatives: Should have predicted this class but didn't
                fn = sum(cm[event_type][pred] if pred in cm[event_type] else 0 
                         for pred in set().union(*[set(preds.keys()) for preds in cm.values()]) 
                         if pred != event_type)
                
                # Calculate metrics
                if tp + fp > 0:
                    precision[event_type] = tp / (tp + fp)
                else:
                    precision[event_type] = 0
                    
                if tp + fn > 0:
                    recall[event_type] = tp / (tp + fn)
                else:
                    recall[event_type] = 0
                    
                if precision[event_type] + recall[event_type] > 0:
                    f1[event_type] = 2 * precision[event_type] * recall[event_type] / (precision[event_type] + recall[event_type])
                else:
                    f1[event_type] = 0
            
            # Calculate macro average metrics
            metrics[model]['precision'] = precision
            metrics[model]['recall'] = recall
            metrics[model]['f1'] = f1
            metrics[model]['macro_precision'] = sum(precision.values()) / len(precision) if precision else 0
            metrics[model]['macro_recall'] = sum(recall.values()) / len(recall) if recall else 0
            metrics[model]['macro_f1'] = sum(f1.values()) / len(f1) if f1 else 0

    return metrics

# -----------------------------------------------------------------------------
# Output and Visualization Functions
# -----------------------------------------------------------------------------
def print_and_write_results(best_params_by_event, metrics, output_file="model_results.txt"):
    """
    Print results to console and write to file.
    
    Args:
        best_params_by_event: Dictionary of best parameters by model and event type
        metrics: Dictionary of metrics by model
        output_file: Name of output file for results
    """
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    # Calculate total event counts by event type
    event_class_totals = defaultdict(int)
    
    # Count total events for each class across all models and parameters
    for model in correct_predictions:
        for params in total_predictions[model]:
            for event_type in event_types:
                event_class_totals[event_type] += total_predictions[model][params][event_type]
    
    # Open file for writing
    with open(output_path, 'w') as f:
        # Function to write both to console and file
        def write_line(line):
            print(line)
            f.write(line + '\n')
        
        # Print total events across all models and parameters
        total_all_events = sum(event_class_totals.values())
        write_line(f"\n=== Total Events: {total_all_events} ===")
        for event_type in event_types:
            write_line(f"Event type: {event_type.upper()}: {event_class_totals[event_type]} total events")
        write_line("")
            
        # Print results by model and event type
        write_line("\n=== Best Parameters by Model and Event Type ===")
        for model in sorted(best_params_by_event.keys()):
            for event_type in event_types:
                if event_type in best_params_by_event[model] and best_params_by_event[model][event_type]:
                    for params in best_params_by_event[model][event_type]:
                        correct = correct_predictions[model][params][event_type]
                        total = total_predictions[model][params][event_type]
                        percent = (correct / total * 100) if total > 0 else 0
                        write_line(f"Model: {model}  event type: {event_type}   params: {params}  "
                                  f"correct predictions: {correct}  total predictions: {total}  "
                                  f"percent correct: {percent:.1f}%.")
            write_line("")

        # Print best overall params for each model
        write_line("\n=== Best Overall Parameters for Each Model ===")
        for model in sorted(metrics.keys()):
            m = metrics[model]
            write_line(f"Model: {model}  params: {m['params']}  "
                      f"correct predictions: {m['correct']}  total predictions: {m['total']}  "
                      f"percent correct: {m['accuracy']*100:.1f}%.")
            write_line(f"  Precision: {m['macro_precision']:.3f}, Recall: {m['macro_recall']:.3f}, "
                      f"F1: {m['macro_f1']:.3f}, Accuracy: {m['accuracy']:.3f}, Total events: {m['total']}")
            write_line(f"  Mean seismogram count: {m['mean_seismogram_count']:.1f}, "
                      f"Median seismogram count: {m['median_seismogram_count']:.1f}")
            
            # Print per-class metrics with total event counts for each class
            for event_type in event_types:
                if event_type in m['precision']:
                    # Get the total event count for this model, parameter set, and event type
                    events_count = total_predictions[model][m['params']][event_type]
                    write_line(f"  - {event_type.upper()}: Precision: {m['precision'][event_type]:.3f}, "
                              f"Recall: {m['recall'][event_type]:.3f}, F1: {m['f1'][event_type]:.3f}, "
                              f"Total events: {events_count}")
            write_line("")
            
    print(f"Results written to {output_path}")

def generate_confusion_matrices(metrics):
    """
    Generate and save confusion matrix plots for each model.
    
    Args:
        metrics: Dictionary of metrics by model
    """
    print("\nGenerating confusion matrices...")
    
    for model in sorted(metrics.keys()):
        # Create the confusion matrix figure
        plt.figure(figsize=(10, 8))
        
        # Extract confusion matrix data
        cm = metrics[model]['confusion_matrix']
        
        # Determine all classes present in the confusion matrix
        all_classes = set(event_types + ['no'])
        for true_label in cm:
            all_classes.add(true_label)
            for pred_label in cm[true_label]:
                all_classes.add(pred_label)
        
        all_classes = sorted(list(all_classes))
        matrix_size = len(all_classes)
        
        # Create numpy array for the confusion matrix
        confusion_array = np.zeros((matrix_size, matrix_size))
        
        # Fill the confusion matrix
        for i, true_label in enumerate(all_classes):
            for j, pred_label in enumerate(all_classes):
                if true_label in cm and pred_label in cm[true_label]:
                    confusion_array[i, j] = cm[true_label][pred_label]
        
        # Plot the confusion matrix
        plt.imshow(confusion_array, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {model} with {metrics[model]["params"]}')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(matrix_size)
        plt.xticks(tick_marks, all_classes, rotation=45)
        plt.yticks(tick_marks, all_classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add the numbers
        thresh = confusion_array.max() / 2.
        for i in range(matrix_size):
            for j in range(matrix_size):
                plt.text(j, i, format(int(confusion_array[i, j]), 'd'),
                         ha="center", va="center",
                         color="white" if confusion_array[i, j] > thresh else "black")
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{model}.png')
        plt.savefig(output_path, dpi=OUTPUT_DPI)
        print(f"Saved confusion matrix for {model} to {output_path}")

def generate_performance_plots(metrics):
    """
    Generate and save performance metric plots (precision, recall, F1).
    
    Args:
        metrics: Dictionary of metrics by model
    """
    print("\nGenerating performance metric plots...")
    
    # Extract data for plotting
    models = sorted(metrics.keys())
    
    # Set up a figure with 3 subplots for precision, recall, and F1
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Set bar width and positions
    bar_width = 0.2
    x = np.arange(len(models))
    
    # Colors for each event type
    colors = {'eq': 'blue', 'ex': 'red', 'su': 'green'}
    
    # Plot precision (top subplot)
    ax = axes[0]
    for i, event_type in enumerate(event_types):
        values = [metrics[model]['precision'][event_type] if event_type in metrics[model]['precision'] else 0 
                  for model in models]
        ax.bar(x + (i-1)*bar_width, values, bar_width, label=event_type.upper(), color=colors[event_type])
    
    ax.set_ylabel('Precision')
    ax.set_title('Precision by Model and Event Type')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot recall (middle subplot)
    ax = axes[1]
    for i, event_type in enumerate(event_types):
        values = [metrics[model]['recall'][event_type] if event_type in metrics[model]['recall'] else 0 
                  for model in models]
        ax.bar(x + (i-1)*bar_width, values, bar_width, label=event_type.upper(), color=colors[event_type])
    
    ax.set_ylabel('Recall')
    ax.set_title('Recall by Model and Event Type')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot F1 (bottom subplot)
    ax = axes[2]
    for i, event_type in enumerate(event_types):
        values = [metrics[model]['f1'][event_type] if event_type in metrics[model]['f1'] else 0 
                  for model in models]
        ax.bar(x + (i-1)*bar_width, values, bar_width, label=event_type.upper(), color=colors[event_type])
    
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Model and Event Type')
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'performance_metrics.png')
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    print(f"Saved performance metrics plot to {output_path}")
    
    # Create a macro-metrics comparison plot
    plt.figure(figsize=(12, 6))
    
    # Width and positions for grouped bars
    bar_width = 0.25
    x = np.arange(len(models))
    
    # Plot macro metrics
    plt.bar(x - bar_width, [metrics[model]['macro_precision'] for model in models], 
            bar_width, label='Precision', color='skyblue')
    plt.bar(x, [metrics[model]['macro_recall'] for model in models], 
            bar_width, label='Recall', color='lightgreen')
    plt.bar(x + bar_width, [metrics[model]['macro_f1'] for model in models], 
            bar_width, label='F1', color='salmon')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Macro-averaged Performance Metrics by Model')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'macro_metrics.png')
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    print(f"Saved macro metrics plot to {output_path}")

def generate_magnitude_histograms(best_overall_params):
    """
    Generate magnitude histograms for correct vs. incorrect predictions by model.
    
    Creates two histograms for each model - one linear scale and one log scale.
    Ensures all histograms use:
    - The same x-axis limits across all models
    - The same y-axis limits across ALL histograms
    - Includes performance metrics on each plot
    """
    print("\nGenerating magnitude histograms...")
    
    # First determine global min and max magnitude across all models
    global_min_mag = float('inf')
    global_max_mag = float('-inf')
    
    # Also track max count for all histograms to standardize y-axes
    global_max_count = 0
    
    # Compute temporary histograms to find the max counts
    temp_hists = {}
    
    # Find the global min and max magnitude across all models
    for model in sorted(model_results.keys()):
        for status in ["correct", "incorrect"]:
            for event_type in event_types:
                if event_type in model_results[model][status] and model_results[model][status][event_type]:
                    mags = model_results[model][status][event_type]
                    if mags:
                        global_min_mag = min(global_min_mag, min(mags))
                        global_max_mag = max(global_max_mag, max(mags))
    
    # If no data found, exit
    if global_min_mag == float('inf') or global_max_mag == float('-inf'):
        print("No magnitude data found for any model. Skipping histogram generation.")
        return
    
    # Calculate global bin parameters
    bins = np.arange(global_min_mag - BIN_WIDTH/2, global_max_mag + BIN_WIDTH, BIN_WIDTH)
    
    # Pre-compute histograms to find maximum y values across ALL histograms
    for model in sorted(model_results.keys()):
        temp_hists[model] = {"correct": {}, "incorrect": {}}
        
        # Skip if no magnitude data for this model
        if not any(model_results[model]["correct"].values()) and not any(model_results[model]["incorrect"].values()):
            continue
        
        # Calculate histograms for each event type
        for status in ["correct", "incorrect"]:
            for event_type in event_types:
                if event_type in model_results[model][status] and model_results[model][status][event_type]:
                    hist, _ = np.histogram(model_results[model][status][event_type], bins=bins)
                    temp_hists[model][status][event_type] = hist
                    
                    # Update global max count for ALL histograms
                    global_max_count = max(global_max_count, np.max(hist) if hist.size > 0 else 0)
    
    # Add a small buffer to max count to prevent bars from touching the top
    global_max_count = int(global_max_count * 1.1) + 1
    
    print(f"Global y-axis maximum for ALL histograms: {global_max_count}")
    
    # Find best parameter metrics for each model
    model_metrics = {}
    for model in sorted(model_results.keys()):
        if model not in best_overall_params:
            continue
        
        params = best_overall_params[model]
        if not params:
            continue
            
        # Calculate metrics
        correct = overall_correct[model][params]
        total = overall_total[model][params]
        accuracy = correct / total if total > 0 else 0
        
        # Calculate precision, recall, F1 for each class
        precision = {}
        recall = {}
        f1 = {}
        class_totals = {}
        
        cm = confusion_matrices[model][params]
        for event_type in event_types:
            # True positives: Predicted this class correctly
            tp = cm[event_type][event_type] if event_type in cm and event_type in cm[event_type] else 0
            
            # False positives: Predicted this class but was wrong
            fp = sum(cm[true][event_type] if event_type in cm[true] else 0 
                     for true in cm if true != event_type)
            
            # False negatives: Should have predicted this class but didn't
            fn = sum(cm[event_type][pred] if pred in cm[event_type] else 0 
                     for pred in set().union(*[set(preds.keys()) for preds in cm.values()]) 
                     if pred != event_type)
            
            # Get total events for this class
            class_totals[event_type] = total_predictions[model][params][event_type]
            
            # Calculate metrics
            if tp + fp > 0:
                precision[event_type] = tp / (tp + fp)
            else:
                precision[event_type] = 0
                
            if tp + fn > 0:
                recall[event_type] = tp / (tp + fn)
            else:
                recall[event_type] = 0
                
            if precision[event_type] + recall[event_type] > 0:
                f1[event_type] = 2 * precision[event_type] * recall[event_type] / (precision[event_type] + recall[event_type])
            else:
                f1[event_type] = 0
        
        # Calculate macro average metrics
        macro_precision = sum(precision.values()) / len(precision) if precision else 0
        macro_recall = sum(recall.values()) / len(recall) if recall else 0
        macro_f1 = sum(f1.values()) / len(f1) if f1 else 0
        
        # Store metrics for this model
        model_metrics[model] = {
            'params': params,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'total': total,
            'class_precision': precision,
            'class_recall': recall, 
            'class_f1': f1,
            'class_totals': class_totals
        }
    
    # Now generate histograms for each model using the global ranges
    for model in sorted(model_results.keys()):
        # Skip if no magnitude data for this model
        if not any(model_results[model]["correct"].values()) and not any(model_results[model]["incorrect"].values()):
            print(f"Skipping {model} - no magnitude data available")
            continue
        
        # Generate both linear and log scale plots
        for scale in ['linear', 'log']:
            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Plot correct predictions (left subplot)
            ax1.set_title(f'Correct Predictions - {model}')
            for event_type in event_types:
                if event_type in model_results[model]["correct"] and model_results[model]["correct"][event_type]:
                    ax1.hist(model_results[model]["correct"][event_type], bins=bins, alpha=0.6, 
                            color=COLORS[event_type], label=f"{event_type.upper()} (n={len(model_results[model]['correct'][event_type])})")
            
            ax1.set_xlabel('Magnitude')
            ax1.set_ylabel('Frequency')
            ax1.set_yscale(scale)
            # Set consistent x-axis limits
            ax1.set_xlim(global_min_mag, global_max_mag)
            # Set consistent y-axis limits for ALL histograms
            if scale == 'linear':
                ax1.set_ylim(0, global_max_count)
            else:  # log scale
                ax1.set_ylim(0.5, global_max_count * 2)  # Start at 0.5 for log scale
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Plot incorrect predictions (right subplot)
            ax2.set_title(f'Incorrect Predictions - {model}')
            for event_type in event_types:
                if event_type in model_results[model]["incorrect"] and model_results[model]["incorrect"][event_type]:
                    ax2.hist(model_results[model]["incorrect"][event_type], bins=bins, alpha=0.6,
                            color=COLORS[event_type], label=f"{event_type.upper()} (n={len(model_results[model]['incorrect'][event_type])})")
            
            ax2.set_xlabel('Magnitude')
            ax2.set_ylabel('Frequency')
            ax2.set_yscale(scale)
            # Set consistent x-axis limits
            ax2.set_xlim(global_min_mag, global_max_mag)
            # Set consistent y-axis limits for ALL histograms (same as the correct prediction plot)
            if scale == 'linear':
                ax2.set_ylim(0, global_max_count)
            else:  # log scale
                ax2.set_ylim(0.5, global_max_count * 2)  # Start at 0.5 for log scale
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # Add the performance metrics as text
            if model in model_metrics:
                m = model_metrics[model]
                metrics_text = (
                    f"Model: {model}  params: {m['params']}\n"
                    f"Precision: {m['macro_precision']:.3f}, Recall: {m['macro_recall']:.3f}, "
                    f"F1: {m['macro_f1']:.3f}, Accuracy: {m['accuracy']:.3f}, Total events: {m['total']}\n"
                )
                
                # Add per-class metrics
                class_metrics = []
                for event_type in event_types:
                    if event_type in m['class_precision']:
                        class_metrics.append(
                            f"{event_type.upper()}: P={m['class_precision'][event_type]:.3f}, "
                            f"R={m['class_recall'][event_type]:.3f}, F1={m['class_f1'][event_type]:.3f}, "
                            f"Events={m['class_totals'][event_type]}"
                        )
                
                if class_metrics:
                    metrics_text += "\n" + "\n".join(class_metrics)
                
                # Add metrics text to the bottom of the figure
                plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=9,
                          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            plt.suptitle(f'Magnitude Distribution for {model} ({scale} scale)')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for metrics text
            
            output_path = os.path.join(OUTPUT_DIR, f'magnitude_histograms_{model}_{scale}.png')
            plt.savefig(output_path, dpi=OUTPUT_DPI)
            print(f"Saved magnitude histogram for {model} ({scale} scale) to {output_path}")


def build_magnitude_histograms(raw_event_data, best_overall_params):
    """
    Build magnitude histograms using only data from the best parameter set for each model.
    
    Args:
        raw_event_data: Dictionary of raw event data
        best_overall_params: Dictionary mapping model -> best parameter set
    """
    print("\nBuilding magnitude histograms for best parameter sets...")

    # Clear any existing model results
    global model_results
    model_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Process each event's data
    for evid, event_entries in raw_event_data.items():
        for entry in event_entries:
            model = entry["model"]
            params = entry["params"]
            pred = entry["pred"]
            analyst = entry["analyst"] 
            magnitude = entry["magnitude"]   

            # Only use data from the best parameter set for each model
            if model in best_overall_params and params == best_overall_params[model]:
                # Handle "no" predictions as incorrect
                if pred == "no":
                    model_results[model]["incorrect"][analyst].append(magnitude)
                # Handle normal predictions
                elif pred == analyst:
                    model_results[model]["correct"][pred].append(magnitude)
                else:
                    model_results[model]["incorrect"][analyst].append(magnitude)

    # Print summary of histogram data
    for model in model_results:
        correct_count = sum(len(mags) for mags in model_results[model]["correct"].values())
        incorrect_count = sum(len(mags) for mags in model_results[model]["incorrect"].values())
        print(f"  {model}: {correct_count} correct predictions, {incorrect_count} incorrect predictions for histogram")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    """Main function to orchestrate the analysis process."""
    # Ensure output directory exists
    ensure_output_dir()
    
    # Process model outputs and get raw event data
    raw_event_data = process_model_outputs()
    
    # Find best parameters
    best_params_by_event, best_overall_params = find_best_parameters()
    
    # Build magnitude histograms using only the best parameter set for each model
    build_magnitude_histograms(raw_event_data, best_overall_params)
    
    # Calculate metrics
    metrics = calculate_metrics(best_overall_params)
    
    # Output results
    print_and_write_results(best_params_by_event, metrics)
    
    # Generate visualizations
    generate_confusion_matrices(metrics)
    generate_performance_plots(metrics)
    generate_magnitude_histograms(best_overall_params)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()


