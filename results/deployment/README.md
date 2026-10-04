# Deployment and accuracy assessment

> Note: the `*_map_sam3.geojson` files and `tile_masks/sam3` outputs currently in this folder were produced with the deprecated `checkpoint_10.pt` (epoch 10) and must be regenerated with `model_best.pt` (epoch 19). The Detectron2 and Matterport maps were produced with the earlier thresholds (0.35 and 0.50) and must also be regenerated.