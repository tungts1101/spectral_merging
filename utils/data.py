import numpy as np
import os
import gdown
import zipfile
import kagglehub
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

# CUB, ImageNet-R, ImageNet-A, OmnibenchMark and VTAB are the versions defined at https://github.com/zhoudw-zdw/RevisitingCIL from here: 
#   @article{zhou2023revisiting,
#        author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
#        title = {Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need},
#        journal = {arXiv preprint arXiv:2303.07338},
#        year = {2023}
#    }

DATA_ROOT = "/home/lis/data"
os.makedirs(DATA_ROOT, exist_ok=True)

def download_and_extract_dataset(dataset_name, file_id, train_subdir="train", test_subdir="test"):
    train_dir = f"{DATA_ROOT}/{dataset_name}/{train_subdir}/"
    test_dir = f"{DATA_ROOT}/{dataset_name}/{test_subdir}/"
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"{dataset_name} dataset not found. Downloading...")
        
        download_url = f"https://drive.google.com/uc?id={file_id}"
        zip_path = f"{DATA_ROOT}/{dataset_name}.zip"
        
        print(f"Downloading {dataset_name} dataset...")
        try:
            gdown.download(download_url, zip_path, quiet=False)
            print("Download completed.")
        except Exception as e:
            raise Exception(f"Failed to download {dataset_name} dataset: {str(e)}")
        
        print(f"Extracting {dataset_name} dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_ROOT)
        
        os.remove(zip_path)
        print(f"{dataset_name} dataset extracted successfully.")
    
    return train_dir, test_dir

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train, args,isCifar=False):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        if isCifar:
            size = input_size
        else:
            size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    use_path = False

    train_trsf=build_transform(True, None,True)
    test_trsf=build_transform(False, None,True)
    common_trsf = [
        # transforms.ToTensor(),
    ]

    class_order = np.arange(100).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        do_download=True
        if os.path.isfile(f"{DATA_ROOT}/cifar-100-python/train"):
            do_download=False
        train_dataset = datasets.cifar.CIFAR100(f"{DATA_ROOT}/", train=True, download=do_download)
        test_dataset = datasets.cifar.CIFAR100(f"{DATA_ROOT}", train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iTinyImageNet(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        train_dir = f"{DATA_ROOT}/tiny-imagenet-200/train/"
        test_dir = f"{DATA_ROOT}/tiny-imagenet-200/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        

class iPlaces365(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        train_dir = f"{DATA_ROOT}/places365_standard/train/"
        test_dir = f"{DATA_ROOT}/places365_standard/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023), download from https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW
        train_dir, test_dir = download_and_extract_dataset("imagenet-r", "1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023), download from  https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL
        train_dir, test_dir = download_and_extract_dataset("imagenet-a", "19l52ua_vvTtttgVRziCZJjal0TPE9f2p")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class CUB(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023) download from https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc
        train_dir, test_dir = download_and_extract_dataset("cub", "1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class omnibenchmark(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023), download from https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA
        train_dir, test_dir = download_and_extract_dataset("omnibenchmark", "1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class vtab(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2013), download from https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EQyTP1nOIH5PrfhXtpPgKQ8BlEFW2Erda1t7Kdi3Al-ePw?e=Yt4RnV
        train_dir, test_dir = download_and_extract_dataset("vtab", "1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        #print(train_dset.class_to_idx)
        #print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class cars(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(196).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        stanford_cars_dir = f"{DATA_ROOT}/stanford_cars"
        
        if not os.path.exists(stanford_cars_dir):
            print("Stanford Cars dataset not found. Downloading from Kaggle...")
            
            try:
                path = kagglehub.dataset_download("rickyyyyyyy/torchvision-stanford-cars")
                print("Path to dataset files:", path)
                
                stanford_cars_source = os.path.join(path, "stanford_cars")
                
                if os.path.exists(stanford_cars_source):
                    import shutil
                    shutil.copytree(stanford_cars_source, stanford_cars_dir)
                    print(f"Successfully copied stanford_cars folder to {stanford_cars_dir}")
                else:
                    import shutil
                    shutil.copytree(path, stanford_cars_dir)
                    print(f"Successfully copied entire dataset folder to {stanford_cars_dir}")
                    
            except Exception as e:
                raise Exception(f"Failed to download Stanford Cars dataset: {str(e)}")

        train_dataset = datasets.StanfordCars(DATA_ROOT, split='train', download=False)
        test_dataset = datasets.StanfordCars(DATA_ROOT, split='test', download=False)
        self.train_data, self.train_targets = split_images_labels(train_dataset._samples)
        self.test_data, self.test_targets = split_images_labels(test_dataset._samples)

class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm=True):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        train_dir = f"{DATA_ROOT}/objectnet/train/"
        test_dir = f"{DATA_ROOT}/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class core50(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def __init__(self,inc,use_input_norm):
        self.inc=inc
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        #download from here: http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz
        train_dir = f"{DATA_ROOT}/core50_imgs/"+self.inc+"/"
        #print(train_dir)
        test_dir = f"{DATA_ROOT}/core50_imgs/test_3_7_10/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class cddb(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(2).tolist()

    def __init__(self,inc,use_input_norm):
        self.inc=inc
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        #download from here: https://coral79.github.io/CDDB_web/
        train_dir = f"{DATA_ROOT}/CDDB/"+self.inc+"/train/"
        #print(train_dir)
        test_dir = f"{DATA_ROOT}/CDDB-hard_val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class domainnet(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(345).tolist()

    def __init__(self,inc,use_input_norm):
        self.inc=inc
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        #download from http://ai.bu.edu/M3SDA/#dataset (use "cleaned version")
        aa=np.loadtxt(f"{DATA_ROOT}/domainnet/"+self.inc+'_train.txt',dtype='str')
        self.train_data=np.array([f"{DATA_ROOT}/domainnet/"+x for x in aa[:,0]])
        self.train_targets=np.array([int(x) for x in aa[:,1]])

        dil_tasks=['real','quickdraw','painting','sketch','infograph','clipart']
        files=[]
        labels=[]
        for task in dil_tasks:
            aa=np.loadtxt(f"{DATA_ROOT}/domainnet/"+task+'_test.txt',dtype='str')
            files+=list(aa[:,0])
            labels+=list(aa[:,1])
        self.test_data=np.array([f"{DATA_ROOT}/domainnet/"+x for x in files])
        self.test_targets=np.array([int(x) for x in labels])

