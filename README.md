# HumanRetrieval
Truy xuất người dựa trên giới tính, loại quần áo và màu sắc

# Cài đặt thư viện
```
pip install -r requirements.txt
# Dùng torchreid
pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip 
```

# Cài đặt DCNv2 dùng YOLACT++
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
%cd ../HumanRetrieval/Detection/yolact/external/DCNv2

# Build DCNv2
python setup.py build develop
```

# File weights

MỘt số file weights : YOLACT++, YOLOR, YOLOv5, EfficientNet     
![link drive](https://drive.google.com/drive/folders/1UVwUD7kz1qlpq9OeMjFOXhCSEfmaznKL?usp=sharing)


# Chạy
```
%cd ../HumanRetrieval/
!python person_retrieval_system_v3.py --source=/content/video212.jpg --device=cuda --top=tee,blue --bottom=trousers,white --humans=male --clothes=short_sleeved_shirt,trousers --extractor=efficientnet-b2 --type_clothes_weight=/content/gdrive/MyDrive/model/EffNet_B2_type_Aug/efficientnet-b2type_clothes.pt --color_clothes_weight=/content/gdrive/MyDrive/model/EffNet_B2_color_Aug/efficientnet-b2color_clothes.pt
```


