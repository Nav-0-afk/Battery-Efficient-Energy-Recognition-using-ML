import numpy as np
import joblib

class EdgeInferencePipeline:
    def __init__(self):
        self.scaler = joblib.load('artifacts/scaler.joblib')
        self.indices = joblib.load('artifacts/selected_indices.joblib')
        
        self.stage0_if = joblib.load('models/stage0_if.joblib')
        self.stage1_lr = joblib.load('models/stage1_lr.joblib')
        self.stage1b_svm = joblib.load('models/stage1b_svm.joblib')
        self.stage2_lgbm = joblib.load('models/stage2_lgbm.joblib')
        
        self.activity_map = {
            0: "WALKING", 1: "WALKING_UPSTAIRS", 2: "WALKING_DOWNSTAIRS",
            3: "SITTING", 4: "STANDING", 5: "LAYING"
        }

    def process_payload(self, raw_sensor_array_561):
        scaled_array = self.scaler.transform(raw_sensor_array_561.reshape(1, -1))
        reduced_array = scaled_array[:, self.indices]
        
        # --- STAGE 0: Anomaly Detection ---
        if self.stage0_if.predict(reduced_array)[0] == -1:
            return "HALT: Anomaly/Device Drop Detected"
            
        # --- STAGE 1: Static vs Dynamic Gate ---
        is_dynamic = self.stage1_lr.predict(reduced_array)[0]
        
        if is_dynamic == 0:
            # --- STAGE 1B: Static Specialist ---
            static_code = self.stage1b_svm.predict(reduced_array)[0]
            activity_name = self.activity_map.get(static_code, "UNKNOWN STATIC")
            return f"STATIC ROUTE -> {activity_name} (LightGBM Bypassed)"
        else:
            # --- STAGE 2: Dynamic Specialist ---
            dynamic_code = self.stage2_lgbm.predict(reduced_array)[0]
            activity_name = self.activity_map.get(dynamic_code, "UNKNOWN DYNAMIC")
            return f"DYNAMIC ROUTE -> {activity_name}"

if __name__ == "__main__":
    pipeline = EdgeInferencePipeline()
    dummy_payload = np.random.rand(561)
    print(pipeline.process_payload(dummy_payload))