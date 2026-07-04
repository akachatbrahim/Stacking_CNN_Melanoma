#project to path
path_project = "/home/akachat/tf_env/Stacking_CNN_Melanoma"
# Training settings
train_dir = f"{path_project}/data/train"
test_dir = f"{path_project}/data/test"
save_model_path = f"{path_project}/Models/Sens95_XGBoost_E40_T0.5/"
#PLOTS_DIR = f"{path_project}/plots/"
num_classes = 1

# ============================================================================
# LEARNING RATE AND WEIGHT DECAY AND OTHER CONFIGURATION
# ============================================================================

learning_rate = 3e-5  
weight_decay = 1e-5 
patience = 10
min_delta = 0.001
batch_size = 32
img_size = (224, 224)
seed = 42
input_shape = (224, 224, 3)
epochs = 40  
number_aug = 1
threshold = 0.5
youden_index_or_sens_sup_95 = "sens_sup_95"  # or "youden_index"
metrics_per_model = {}
ml_models = {}
