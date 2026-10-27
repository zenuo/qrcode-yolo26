# 三类圆码：种子合成、网络采集、自动打标流程使用
CLASS_NAMES = ("alipay", "wechat_miniprogram", "douyin")
# 训练数据集 / 模型的五类，与 dataset/data.yaml 一致；前三类 id 与 CLASS_NAMES 相同
DATASET_CLASS_NAMES = ("alipay", "wechat", "douyin", "barcode", "qrcode")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {index: name for name, index in CLASS_TO_ID.items()}
NEGATIVE_CLASS = "negative"

SEED_DIR_NAME = "raw/seeds"
DEFAULT_IMAGE_SIZE = 640
