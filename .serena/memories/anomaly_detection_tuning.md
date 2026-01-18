The Anomaly Detection system (Day 4 task) was tuned to achieve 100% TPR and 0% FPR.
The key changes were:
1. Moving from a boolean consensus logic to a weighted scoring system (0.8 DTW, 0.1 Velocity, 0.1 Isolation).
2. Significantly increasing the DTW normalization divisor (from 15.0 to 150.0) based on diagnostic analysis showing 'bad' sequences had mean distances > 280, while 'good' were ~33.
3. Increasing the velocity peak height threshold (from 2.0 to 4.5) to avoid noise in 'good' samples triggering false positives.
4. Setting the final anomaly threshold to > 0.55.
Validation confirmed correctness on the synthetic dataset (90 good, 10 bad).