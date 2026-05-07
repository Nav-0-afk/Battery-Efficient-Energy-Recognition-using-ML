import numpy as np
import onnxruntime as ort
import time
import joblib

def benchmark_isolated_hardware_latency():
    # 1. Setup Session Options to minimize overhead
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    sessions = {
        "Stage 0 (IF)": ort.InferenceSession("onnx_models/stage0_if.onnx", sess_options=opts),
        "Stage 1 (LR)": ort.InferenceSession("onnx_models/stage1_lr.onnx", sess_options=opts),
        "Stage 1B (SVM)": ort.InferenceSession("onnx_models/stage1b_svm.onnx", sess_options=opts),
        "Stage 2 (XGB)": ort.InferenceSession("onnx_models/stage2_xgb.onnx", sess_options=opts)
    }

    # Load 1 single row of test data
    indices = joblib.load('artifacts/selected_indices.joblib')
    X_test = np.load('artifacts/X_test_scaled.npy')[:, indices]
    dummy_input = X_test[0:1].astype(np.float32)

    print("EXECUTION LATENCY")
    
    raw_latencies = {}

    for name, session in sessions.items():
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(100):
            session.run(None, {input_name: dummy_input})
            
        # Timed Execution Loop (1000 iterations inside the model)
        start = time.perf_counter()
        for _ in range(1000):
            session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        
        # Calculate isolated microsecond latency
        lat = ((end - start) / 1000) * 1_000_000
        raw_latencies[name] = lat
        print(f"{name:<15}: {lat:.2f} µs")

    # Calculate Total Pipeline Paths
    path_a = raw_latencies["Stage 0 (IF)"] + raw_latencies["Stage 1 (LR)"] + raw_latencies["Stage 2 (XGB)"]
    path_b = raw_latencies["Stage 0 (IF)"] + raw_latencies["Stage 1 (LR)"] + raw_latencies["Stage 1B (SVM)"]
    
    print("THEORETICAL LATENCY")
    print(f"Path A (Normal -> Gate -> XGB) : {path_a:.2f} µs")
    print(f"Path B (Normal -> Gate -> SVM) : {path_b:.2f} µs")

if __name__ == "__main__":
    benchmark_isolated_hardware_latency()