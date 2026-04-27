import numpy as np
import joblib

class EdgeInferencePipeline:
    def __init__(self):
        # Load artifacts
        self.scaler = joblib.load('artifacts/scaler.joblib')
        self.indices = joblib.load('artifacts/selected_indices.joblib')
        
        # Load the 3-Tier Models
        self.stage0_if = joblib.load('models/stage0_if.joblib') # NEW
        self.stage1_svm = joblib.load('models/stage1_svm.joblib')
        self.stage2_xgb = joblib.load('models/stage2_xgb.joblib')
        
        self.activity_map = {
            0: "WALKING", 1: "WALKING_UPSTAIRS", 2: "WALKING_DOWNSTAIRS",
            3: "SITTING", 4: "STANDING", 5: "LAYING"
        }

    def process_payload(self, raw_sensor_array_561):
        # 1. Scale and Slice
        scaled_array = self.scaler.transform(raw_sensor_array_561.reshape(1, -1))
        reduced_array = scaled_array[:, self.indices]
        
        # --- STAGE 0: The Anomaly Gate ---
        is_normal = self.stage0_if.predict(reduced_array)[0]
        
        if is_normal == -1:
            # The data is outside the bounds of human activity. 
            # Kill the process immediately. Zero battery wasted on ML.
            return "GATE 0 CLOSED: Anomaly/Device Drop Detected"
            
        # --- STAGE 1: The Static/Dynamic Gate ---
        is_dynamic = self.stage1_svm.predict(reduced_array)[0]
        
        if is_dynamic == 0:
            return "GATE 1 CLOSED: Static State Detected"
            
        # --- STAGE 2: The Specialist ---
        specific_activity_code = self.stage2_xgb.predict(reduced_array)[0]
        activity_name = self.activity_map.get(specific_activity_code, "UNKNOWN")
        return f"GATE 2 OPENED: Dynamic Activity -> {activity_name}"

if __name__ == "__main__":
    pipeline = EdgeInferencePipeline()
    
    # Test 1: Simulating an anomaly (e.g., dropping the device yields huge sensor spikes)
    anomaly_payload = np.random.rand(561) * 100 
    print("Testing Anomaly:", pipeline.process_payload(anomaly_payload))