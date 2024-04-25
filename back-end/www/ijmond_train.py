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

# Finetune the RGB-TC model from the RGB-I3D model
model.fit(
    p_model="../data/pretrained_models/RGB-I3D-S3.pt",
    model_id_suffix="-s3",
    p_metadata_train="/home/rszabo/uva_thesis_project/data/ijmond_split/metadata_train_split_by_date.json",
    p_metadata_validation="/home/rszabo/uva_thesis_project/data/ijmond_split/metadata_validation_split_by_date.json",
    p_metadata_test="/home/rszabo/uva_thesis_project/data/ijmond_split/metadata_test_split_by_date.json",
    save_model_path="../data/saved_i3d/1/model/")