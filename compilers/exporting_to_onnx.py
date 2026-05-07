import os
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMlFloatTensorType

def export_final_stable_pipeline():
    out_dir = '../onnx_models'
    os.makedirs(out_dir, exist_ok=True)

    indices = joblib.load('../artifacts/selected_indices.joblib')
    num_features = len(indices)
    
    skl_type = [('sensor_input', SklearnFloatTensorType([None, num_features]))]
    xgb_type = [('sensor_input', OnnxMlFloatTensorType([None, num_features]))]
    opset = {'': 15, 'ai.onnx.ml': 3}

    print("--- EXPORTING STABLE FP32 MODELS ---")

    # Stage 0 (Pruned)
    s0 = joblib.load('../models/stage0_if.joblib')
    onnx_s0 = convert_sklearn(s0, initial_types=skl_type, target_opset=opset)
    with open(f'{out_dir}/stage0_if.onnx', "wb") as f: f.write(onnx_s0.SerializeToString())

    # Stage 1 (Standard)
    s1 = joblib.load('../models/stage1_lr.joblib')
    onnx_s1 = convert_sklearn(s1, initial_types=skl_type, target_opset=opset)
    with open(f'{out_dir}/stage1_lr.onnx', "wb") as f: f.write(onnx_s1.SerializeToString())

    # Stage 1B (Standard)
    s1b = joblib.load('../models/stage1b_svm.joblib')
    onnx_s1b = convert_sklearn(s1b, initial_types=skl_type, target_opset=opset)
    with open(f'{out_dir}/stage1b_svm.onnx', "wb") as f: f.write(onnx_s1b.SerializeToString())

    # Stage 2 (Pruned XGBoost)
    s2 = joblib.load('../models/stage2_xgb.joblib')
    onnx_s2 = onnxmltools.convert_xgboost(s2, initial_types=xgb_type, target_opset=15)
    with open(f'{out_dir}/stage2_xgb.onnx', "wb") as f: f.write(onnx_s2.SerializeToString())

    print("All models exported successfully in FP32.")

if __name__ == "__main__":
    export_final_stable_pipeline()