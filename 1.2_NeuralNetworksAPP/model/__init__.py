#__init__.py文件用于标识一个目录为Python包，并且可以包含包的初始化代码，可以为空
from .mtl_cnn import MultiTaskLossWrapper as mtl_cnn
from .mtl_unet_cbam import MultiTaskLossWrapper as mtl_unet_cbam
from .mtl_unet import MultiTaskLossWrapper as mtl_unet
from .xception import MultiTaskLossWrapper as xception