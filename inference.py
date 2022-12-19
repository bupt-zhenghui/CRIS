import cv2
import torch
import utils.config as config
from model import build_segmenter
from utils.dataset import tokenize
import numpy as np
import matplotlib.pyplot as plt


cfg = config.load_cfg_from_cfg_file("./config/refcoco/cris_r50.yaml")
PATH = "exp/refcoco/CRIS_R50/best_model.pth"
model, _ = build_segmenter(cfg)
model = torch.nn.DataParallel(model)

checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()
print("=> loaded checkpoint '{}'".format(PATH))

input_size = (416, 416)


def getTransformMat(img_size, inverse=False):
    ori_h, ori_w = img_size
    inp_h, inp_w = input_size
    scale = min(inp_h / ori_h, inp_w / ori_w)
    new_h, new_w = ori_h * scale, ori_w * scale
    bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

    src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
    dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                    [bias_x, new_h + bias_y]], np.float32)

    mat = cv2.getAffineTransform(src, dst)
    if inverse:
        mat_inv = cv2.getAffineTransform(dst, src)
        return mat, mat_inv
    return mat, None


def convert(img):
    img_size = img.shape[:2]
    mat, mat_inv = getTransformMat(img_size, False)
    img = cv2.warpAffine(
        img,
        mat,
        input_size,
        flags=cv2.INTER_CUBIC,
        borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

    # Image ToTensor & Normalize
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    if not isinstance(img, torch.FloatTensor):
        img = img.float()

    mean = torch.tensor([0.48145466, 0.4578275,
                         0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258,
                        0.27577711]).reshape(3, 1, 1)
    img.div_(255.).sub_(mean).div_(std)
    return img


def inference(img_path, sent):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = convert(img).unsqueeze(0)
    text = tokenize(sent, 17, True)
    # text = text.cuda(non_blocking=True)
    pred = model(img, text)

    pred = torch.sigmoid(pred)
    pred = np.array(pred > 0.35)
    pred = np.array(pred * 255, dtype=np.uint8)
    return pred[0, 0]


if __name__ == '__main__':
    img = cv2.imread("/Users/zhenghui/PycharmProjects/CRIS/exp/refcoco/CRIS_R50/vis/102-img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = convert(img).unsqueeze(0)
    print('img shape: ', img.shape)
    sent = "child sitting on woman's lap"
    text = tokenize(sent, 17, True)
    # text = text.cuda(non_blocking=True)
    pred = model(img, text)
    print(pred.shape)

    pred = torch.sigmoid(pred)
    pred = np.array(pred > 0.35)
    pred = np.array(pred * 255, dtype=np.uint8)
    plt.imshow(pred[0, 0])
    plt.show()
