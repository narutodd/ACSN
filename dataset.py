import os
from os.path import join

import numpy
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import transforms, functional
import torchvision
import random
import SimpleITK as sitk

class Mydataset(data.Dataset):
    def __init__(self, imgpaths_A, imgpaths_B, unaligned=False, resize=False, val=False):
        self.imgpaths_A = imgpaths_A
        self.imgpaths_B = imgpaths_B
        self.unaligned = unaligned
        self.norm = transforms.Normalize((0.5,), (0.5,))
        self.resize = resize
        self.val = val
        if self.resize:
            self.image_resize = transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            self.label_resize = transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.Flip = transforms.RandomHorizontalFlip(1)
    def __getitem__(self, idx):
        imgpath_A = self.imgpaths_A[idx]
        case_root = imgpath_A.split("_0000.nii.gz")[-2]
        case_name = case_root.split("\\")[-1]
        data_root = case_root.split("\\%s" % case_name)[0]
        mode_name = data_root.split("\\")[-1]
        mode = mode_name.split("images")[-1]
        label_root = data_root.split("\\%s" % mode_name)[0]
        labelpath = join(label_root, "labels" + "%s" % mode, case_name + ".nii.gz")
        imgpath_B = join(data_root, case_name + "_0001.nii.gz")

        CT_src = sitk.ReadImage(imgpath_A)
        PET_src = sitk.ReadImage(imgpath_B)
        mask_src = sitk.ReadImage(labelpath)
        CT_array = sitk.GetArrayFromImage(CT_src)
        PET_array = sitk.GetArrayFromImage(PET_src)
        mask_array = sitk.GetArrayFromImage(mask_src)

        CT = self.norm(torch.from_numpy(CT_array).unsqueeze(0))
        PET = self.norm(torch.from_numpy(PET_array).unsqueeze(0))
        mask = torch.from_numpy(mask_array).float().unsqueeze(0)

        if self.val:
            p = 0
        else:
            p = numpy.random.randint(3)
        if p == 1:
            CT = self.Flip(CT)
            PET = self.Flip(PET)
            mask = self.Flip(mask)
        elif p == 2:
            angle = transforms.RandomRotation.get_params([-15, 15])
            CT = functional.rotate(CT, angle)
            PET = functional.rotate(PET, angle)
            mask = functional.rotate(mask, angle)

        if self.resize == True:
            CT = self.image_resize(CT)
            PET = self.image_resize(PET)
            mask = self.label_resize(mask)

        return CT, PET, mask, case_name

    def __len__(self):
        return max(len(self.imgpaths_A), len(self.imgpaths_B))
