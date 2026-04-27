import treelite
import tl2cgen
import os

def compile_model_to_c():
    model_path = 'models/stage2_lgbm_native.txt'
    output_dir = './treelite_c_source'
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Run 3_train_models.py first.")
        return

    print("Loading native LightGBM model into Treelite...")
    # 1. Correct 4.x loader
    model = treelite.frontend.load_lightgbm_model(model_path)
    
    print(f"Exporting raw C code using tl2cgen to {output_dir}/...")
    # 2. Correct 4.x compiler
    tl2cgen.generate_c_code(model, dirpath=output_dir, params={})
    
    print("\nCompilation successful.")
    print(f"Look inside the '{output_dir}' directory.")
    print("You will find 'header.h' and 'main.c' files containing your LightGBM trees written entirely in C.")

if __name__ == "__main__":
    compile_model_to_c()