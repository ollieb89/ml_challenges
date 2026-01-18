"""
Validation Script for Form Anomaly Detection
Generates comprehensive synthetic dataset, runs evaluation, and results in a report.
"""

import sys
import os
import numpy as np
import time
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from pose_analyzer.synthetic_poses import SyntheticPoseGenerator, SquatType
from pose_analyzer.form_anomaly_detector import create_anomaly_detector

def validate_anomaly_detection():
    print("Starting Form Anomaly Detector Validation...")
    
    # 1. Setup Generator
    generator = SyntheticPoseGenerator()
    
    # 2. Generate Reference Template (Good Form, No Noise)
    print("Generating reference template...")
    reference_sequence = generator.generate_squat_sequence(
        num_frames=60, 
        squat_type=SquatType.PARALLEL, 
        noise_level=0.0
    )
    reference_angles = [pose.ground_truth_angles for pose in reference_sequence]
    
    # 3. Create Detector
    detector = create_anomaly_detector(reference_angles)
    
    # 4. Generate Test Datasets
    print("\nGenerating test datasets...")
    
    # Good Form: 90 reps, low noise
    NUM_GOOD = 90
    good_sequences = []
    print(f"Generating {NUM_GOOD} good form sequences...")
    for _ in range(NUM_GOOD):
        seq = generator.generate_squat_sequence(
            num_frames=60,
            squat_type=SquatType.PARALLEL,
            noise_level=0.5
        )
        good_sequences.append(seq)
        
    # Bad Form: 10 reps, high noise (simulating jerky/erratic motion)
    NUM_BAD = 10
    bad_sequences = []
    print(f"Generating {NUM_BAD} bad form sequences...")
    for _ in range(NUM_BAD):
        seq = generator.generate_squat_sequence(
            num_frames=60,
            squat_type=SquatType.PARALLEL,
            noise_level=3.0  # High noise triggers velocity and DTW anomalies
        )
        bad_sequences.append(seq)
    
    # Save dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '../data/validation/labeled_squats.npz')
    # Save a simplified version for storage (just keypoints and labels)
    # Label 0: Good, 1: Bad
    all_sequences_keypoints = []
    all_labels = []
    
    for seq in good_sequences:
        kps = np.array([p.keypoints for p in seq])
        all_sequences_keypoints.append(kps)
        all_labels.append(0)
        
    for seq in bad_sequences:
        kps = np.array([p.keypoints for p in seq])
        all_sequences_keypoints.append(kps)
        all_labels.append(1)
        
    np.savez(dataset_path, 
             keypoints=np.array(all_sequences_keypoints), 
             labels=np.array(all_labels),
             reference=np.array([p.ground_truth_angles for p in reference_sequence], dtype=object) # Pickle object
    )
    print(f"Dataset saved to {dataset_path}")

    # 5. Run Evaluation
    print("\nRunning evaluation...")
    y_true = []
    y_pred = []
    
    # Evaluate Good Sequences
    print("Evaluating Good Form sequences...")
    # metrics for stats
    good_scores = []
    
    for i, seq in enumerate(good_sequences):
        detector.reset() # Reset state for new sequence
        anomalies_in_seq = 0
        max_score = 0.0
        
        for pose in seq:
            result = detector.detect_anomaly(pose.keypoints, pose.confidences)
            if result.is_anomaly:
                anomalies_in_seq += 1
            max_score = max(max_score, result.anomaly_score)
            
        good_scores.append(max_score)
        
        # Criteria: If > 10% frames are anomalous, flag sequence as anomaly
        is_seq_anomaly = anomalies_in_seq > (len(seq) * 0.1)
        
        y_true.append(0) # Good
        y_pred.append(1 if is_seq_anomaly else 0)
        
    # Evaluate Bad Sequences
    print("Evaluating Bad Form sequences...")
    bad_scores = []
    
    for i, seq in enumerate(bad_sequences):
        detector.reset()
        anomalies_in_seq = 0
        max_score = 0.0
        
        for pose in seq:
            result = detector.detect_anomaly(pose.keypoints, pose.confidences)
            if result.is_anomaly:
                anomalies_in_seq += 1
            max_score = max(max_score, result.anomaly_score)

        bad_scores.append(max_score)
            
        is_seq_anomaly = anomalies_in_seq > (len(seq) * 0.1)
        
        y_true.append(1) # Bad
        y_pred.append(1 if is_seq_anomaly else 0)

    # 6. Calculate Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\n=== Validation Results ===")
    print(f"Total Samples: {len(y_true)}")
    print(f"True Positives (Bad flagged Bad): {tp}")
    print(f"False Positives (Good flagged Bad): {fp}")
    print(f"True Negatives (Good flagged Good): {tn}")
    print(f"False Negatives (Bad flagged Good): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (TPR): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    print(f"\nScore Stats:")
    print(f"Avg Max Score (Good): {np.mean(good_scores):.4f}")
    print(f"Avg Max Score (Bad): {np.mean(bad_scores):.4f}")

    # 7. Generate Report
    report_path = os.path.join(os.path.dirname(__file__), '../reports/form_scoring_validation.md')
    
    with open(report_path, 'w') as f:
        f.write("# Form Scorer & Anomaly Detection Validation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write("The Form Anomaly Detector was evaluated on a synthetic dataset of 100 squat sequences (90 Good, 10 Bad).\n")
        f.write("The system achieved:\n")
        f.write(f"- **True Positive Rate (TPR):** {tpr*100:.1f}%\n")
        f.write(f"- **False Positive Rate (FPR):** {fpr*100:.1f}%\n")
        f.write(f"- **F1 Score:** {f1:.4f}\n\n")
        
        success = (tpr >= 0.95) and (fpr <= 0.05)
        f.write(f"**Status:** {'✅ PASSED' if success else '❌ FAILED'}\n")
        f.write("Criteria: TPR >= 95%, FPR <= 5%\n\n")
        
        f.write("## 2. Methodology\n")
        f.write("- **Dataset:** Synthetic 60-frame squat sequences.\n")
        f.write("- **Reference:** Parallel squat, 0.0 noise level.\n")
        f.write("- **Good Form:** 90 samples, Parallel squat, 0.5 noise level.\n")
        f.write("- **Bad Form:** 10 samples, Parallel squat, 3.0 noise level (simulating erratic movement/severe instability).\n")
        f.write("- **Detection Logic:** Weighted combination of DTW, Velocity Peaks, and Isolation Forest.\n")
        f.write("- **Sequence Classification:** Labeled as anomaly if > 10% of frames are anomalous.\n\n")

        f.write("## 3. Detailed Metrics\n")
        f.write("| Metric | Value | Target |\n")
        f.write("|--------|-------|--------|\n")
        f.write(f"| Precision | {precision:.4f} | > 0.9 |\n")
        f.write(f"| Recall | {recall:.4f} | > 0.95 |\n")
        f.write(f"| F1 Score | {f1:.4f} | > 0.9 |\n")
        f.write(f"| FPR | {fpr:.4f} | < 0.05 |\n\n")
        
        f.write("## 4. Confusion Matrix\n")
        f.write("```\n")
        f.write(f"                 Predicted Good   Predicted Bad\n")
        f.write(f"Actual Good      {tn:<14}   {fp:<14}\n")
        f.write(f"Actual Bad       {fn:<14}   {tp:<14}\n")
        f.write("```\n\n")
        
        f.write("## 5. Score Distribution\n")
        f.write(f"- **Good Form Avg Max Score:** {np.mean(good_scores):.4f} (Std: {np.std(good_scores):.4f})\n")
        f.write(f"- **Bad Form Avg Max Score:** {np.mean(bad_scores):.4f} (Std: {np.std(bad_scores):.4f})\n")
        f.write(f"- **Anomaly Threshold:** {detector.anomaly_threshold}\n\n")
        
        f.write("## 6. Conclusions\n")
        if success:
            f.write("The anomaly detector meets the performance requirements for the daily challenge.\n")
            f.write("It successfully distinguishes between normal variation (noise=0.5) and severe form breakdown (noise=3.0).\n")
        else:
            f.write("The anomaly detector requires further tuning.\n")
            if fpr > 0.05:
                f.write("- FPR is too high. Consider increasing the anomaly threshold or the DTW window size.\n")
            if tpr < 0.95:
                f.write("- TPR is too low. The detector is missing bad form. Consider lowering threshold or increasing sensitivity to velocity peaks.\n")
                
    print(f"\nReport generated at {report_path}")

if __name__ == "__main__":
    validate_anomaly_detection()
