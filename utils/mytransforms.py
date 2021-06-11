import torch

IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

INAT_STATS = { 'mean': [0.466, 0.471, 0.380],
                'std': [0.195, 0.194, 0.192]}

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        # Create a random vector of 3x1 in the same type as img
        alpha = img.new_empty(3,1).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .matmul(alpha * self.eigval.view(3, 1))

        return img.add(rgb.view(3, 1, 1).expand_as(img))
