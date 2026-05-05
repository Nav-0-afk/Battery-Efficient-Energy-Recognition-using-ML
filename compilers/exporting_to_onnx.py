import os
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMlFloatTensorType

def export_pipeline_to_onnx():
    out_dir = '../onnx_models'
    os.makedirs(out_dir, exist_ok=True)

    print("--- STARTING PROFESSIONAL ONNX EXPORT ---")

    indices = joblib.load('../artifacts/selected_indices.joblib')
    num_features = len(indices)
    
    skl_initial_type = [('sensor_input', SklearnFloatTensorType([None, num_features]))]
    xgb_initial_type = [('sensor_input', OnnxMlFloatTensorType([None, num_features]))]
    
    # THE FIX: Separate the versioning logic for the two different libraries
    skl_opset = {'': 15, 'ai.onnx.ml': 3}  # skl2onnx expects a dictionary
    xgb_opset = 15                         # onnxmltools strictly expects an integer
    
    print(f"Configuring ONNX for dynamic input shape: [None, {num_features}]")

    print("Loading models from /models directory...")
    stage0_if = joblib.load('../models/stage0_if.joblib')
    stage1_lr = joblib.load('../models/stage1_lr.joblib')
    stage1b_svm = joblib.load('../models/stage1b_svm.joblib')
    stage2_xgb = joblib.load('../models/stage2_xgb.joblib')

    print("Exporting Stage 0 (Isolation Forest) -> ONNX...")
    onnx_if = convert_sklearn(stage0_if, initial_types=skl_initial_type, target_opset=skl_opset)
    with open(f'{out_dir}/stage0_if.onnx', "wb") as f:
        f.write(onnx_if.SerializeToString())

    print("Exporting Stage 1 (Logistic Regression) -> ONNX...")
    onnx_lr = convert_sklearn(stage1_lr, initial_types=skl_initial_type, target_opset=skl_opset)
    with open(f'{out_dir}/stage1_lr.onnx', "wb") as f:
        f.write(onnx_lr.SerializeToString())

    print("Exporting Stage 1B (LinearSVC) -> ONNX...")
    onnx_svm = convert_sklearn(stage1b_svm, initial_types=skl_initial_type, target_opset=skl_opset)
    with open(f'{out_dir}/stage1b_svm.onnx', "wb") as f:
        f.write(onnx_svm.SerializeToString())

    print("Exporting Stage 2 (XGBoost) -> ONNX...")
    # Feed the integer-based xgb_opset here
    onnx_xgb = onnxmltools.convert_xgboost(stage2_xgb, initial_types=xgb_initial_type, target_opset=xgb_opset)
    with open(f'{out_dir}/stage2_xgb.onnx', "wb") as f:
        f.write(onnx_xgb.SerializeToString())

    print("\n--- ONNX EXPORT COMPLETE ---")
    print(f"All models successfully converted to universal hardware format in: {out_dir}")

if __name__ == "__main__":
    export_pipeline_to_onnx()