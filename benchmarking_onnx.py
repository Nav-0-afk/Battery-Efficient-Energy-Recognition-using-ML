import numpy as np
import onnxruntime as ort
import time
import joblib

def benchmark_optimized_architecture():
    # 1. Setup Session Options for Bare-Metal Simulation
    opts = ort.SessionOptions()
    
    # CRITICAL: Disable multi-threading overhead for batch_size=1
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Maximize graph optimizations (constant folding, node fusion)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 2. Initialize Sessions with Optimized Providers
    # Explicitly use the CPU provider to simulate the target wearable environment
    providers = ['CPUExecutionProvider']
    
    sessions = {
        "stage0": ort.InferenceSession("onnx_models/stage0_if.onnx", sess_options=opts, providers=providers),
        "stage1": ort.InferenceSession("onnx_models/stage1_lr.onnx", sess_options=opts, providers=providers),
        "stage1b": ort.InferenceSession("onnx_models/stage1b_svm.onnx", sess_options=opts, providers=providers),
        "stage2": ort.InferenceSession("onnx_models/stage2_xgb.onnx", sess_options=opts, providers=providers)
    }

    # 3. Load Test Data
    indices = joblib.load('artifacts/selected_indices.joblib')
    X_test = np.load('artifacts/X_test_scaled.npy')[:, indices].astype(np.float32)
    samples = X_test[:2000].astype(np.float32)

    def run_inf(session, data):
        return session.run(None, {session.get_inputs()[0].name: data})[0]

    latencies = []
    print(f"--- RUNNING OPTIMIZED PIPELINE BENCHMARK ---")

    for i in range(len(samples)):
        input_row = samples[i:i+1]
        
        start = time.perf_counter()

        # Architecture Logic (0.85 Threshold)
        if run_inf(sessions["stage0"], input_row) == 1:
            probs = run_inf(sessions["stage1"], input_row)
            if np.max(probs) < 0.85:
                _ = run_inf(sessions["stage1b"], input_row)
            else:
                _ = run_inf(sessions["stage2"], input_row)
        
        latencies.append((time.perf_counter() - start) * 1_000_000)

    print(f"\n[OPTIMIZED PERFORMANCE METRICS]")
    print(f"Average Latency: {np.mean(latencies):.2f} µs")
    print(f"95th Percentile: {np.percentile(latencies, 95):.2f} µs")

if __name__ == "__main__":
    benchmark_optimized_architecture()