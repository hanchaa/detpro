_base_ = 'mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_coco_pretrain_ens.py'
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000025)
lr_config = dict(step=[16])
total_epochs = 20
evaluation = dict(interval=2,metric=['bbox', 'segm'])
checkpoint_config = dict(interval=1, create_symlink=False)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2
)
