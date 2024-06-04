# Import I3D
from i3d_learner import I3dLearner

# Initialize the model
model = I3dLearner(
    mode="rgb",
    augment=True,
    p_frame="/projects/0/prjs0930/data/rgb/",
    use_tc=True,
    freeze_i3d=True,
    batch_size_train=8,
    milestones=[1000, 2000],
    num_steps_per_update=1)

# Change this to your split number (can be "s0", "s1", "s2", "s3", "s4", or "s5")
split_str = "0000000-i3d-rgb-s5"

# Make sure that the "paper_result" folder is in the "data" foler
model_root_path = "../data/paper_result/RGB-I3D/" + split_str

# Change this to the correct model number under the "model" folder
p_model = model_root_path + "/model/585.pt"

# No need to change the variables below
model_id_suffix = "-" + split_str
p_metadata_train = "/home/rszabo/uva_thesis_project/data/split/metadata_train_split_4_by_camera.json"
p_metadata_validation = "/home/rszabo/uva_thesis_project/data/split/metadata_validation_split_4_by_camera.json"
p_metadata_test = "/home/rszabo/uva_thesis_project/data/split/metadata_test_split_4_by_camera.json"

# Finetune the RGB-TC model from the RGB-I3D model
model.fit(
    p_model=p_model,
    model_id_suffix=model_id_suffix,
    p_metadata_train=p_metadata_train,
    p_metadata_validation=p_metadata_validation,
    p_metadata_test=p_metadata_test,
    save_model_path="../data/saved_i3d/5_split/model/")