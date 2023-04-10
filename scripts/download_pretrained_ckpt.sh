echo ----- Start downloading clip and segment-anything pretrained model weights
mkdir pretrained
cd pretrained
# download clip model
wget -O ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
# download sam model
wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

