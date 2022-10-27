_base_ = ['./mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_coco_pretrain.py']

checkpoint_config = dict(interval=2,create_symlink=False)
load_from = 'data/current_mmdetection_Head.pth'

total_epochs = 1

model = dict(roi_head=dict(type='StandardRoIHeadColReuse',save_feature_dir='./data/coco_clip_image_proposal_embedding/train/train2017'))
# model = dict(roi_head=dict(save_feature_dir='data/LVIS_prompt_train/train'))

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "zero-shot/instances_train2017_seen_2_oriorder.json",
        proposal_file=None,
        img_prefix=data_root + "train2017/"
    ),
    val=dict(
        proposal_file=None
    )
)
