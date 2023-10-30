import matplotlib.pyplot as plt
import matplotlib as mpl
from importlib import reload
import IPython
mpl.rcParams['lines.linewidth'] = 0.25
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 0.25

import torch, argparse, os, sys, shutil, inspect, json, numpy, math
sys.path.append("/home/yulu_gan/Unified_architecture/dissect")
import netdissect
from netdissect.easydict import EasyDict
from netdissect import pbar, nethook, renormalize, parallelfolder, pidfile
from netdissect import upsample, tally, imgviz, imgsave, bargraph, show
from experiment import dissect_experiment as experiment

# choices are alexnet, vgg16, or resnet152.
args = EasyDict(model='vgg16', dataset='places', seg='netpqc', layer='conv5_3', quantile=0.01)
resdir = 'results/%s-%s-%s-%s-%s' % (args.model, args.dataset, args.seg, args.layer, int(args.quantile * 1000))
def resfile(f):
    return os.path.join(resdir, f)

model = experiment.load_model(args)
layername = experiment.instrumented_layername(args)
model.retain_layer(layername)
dataset = experiment.load_dataset(args)
upfn = experiment.make_upfn(args, dataset, model, layername)
sample_size = len(dataset)
percent_level = 1.0 - args.quantile

print('Inspecting layer %s of model %s on %s' % (layername, args.model, args.dataset))

# Classifier labels
from urllib.request import urlopen
from netdissect import renormalize

# synset_url = 'http://gandissect.csail.mit.edu/models/categories_places365.txt'
# classlabels = [r.split(' ')[0][3:] for r in urlopen(synset_url).read().decode('utf-8').split('\n')]
classlabels = dataset.classes
segmodel, seglabels, segcatlabels = experiment.setting.load_segmenter(args.seg)
renorm = renormalize.renormalizer(dataset, target='zc')

from netdissect import renormalize

indices = [200, 755, 709, 423, 60, 100, 110, 120]
batch = torch.cat([dataset[i][0][None,...] for i in indices])
truth = [classlabels[dataset[i][1]] for i in indices]
preds = model(batch.cuda()).max(1)[1]
imgs = [renormalize.as_image(t, source=dataset) for t in batch]
prednames = [classlabels[p.item()] for p in preds]
for i, (img, pred, gt) in enumerate(zip(imgs, prednames, truth)):
    img.save(f"image_{i}_pred_{pred}_true_{gt}.png")


from netdissect import imgviz

iv = imgviz.ImageVisualizer(120, source=dataset)
seg = segmodel.segment_batch(renorm(batch).cuda(), downsample=4)

for i in range(len(seg)):
    img = iv.image(batch[i])
    seg_img = iv.segmentation(seg[i, 0])
    seg_key_img = iv.segment_key(seg[i, -1], segmodel)

    img.save(f"image_{i}.png")
    seg_img.save(f"segmentation_{i}.png")
    seg_key_img.save(f"segment_key_{i}.png")


from netdissect import imgviz

acts = model.retained_layer(layername).cpu()
ivsmall = imgviz.ImageVisualizer((100, 100), source=dataset)
import os
if not os.path.exists('visualization_images'):
    os.makedirs('visualization_images')

# 保存图像到文件
for u in range(min(acts.shape[1], 12)):
    masked_img = ivsmall.masked_image(batch[0], acts, (0, u), percent_level=0.99)
    heatmap_img = ivsmall.heatmap(acts, (0, u), mode='nearest')
    
    masked_img.save(f"visualization_images/masked_image_unit_{u}.png")
    heatmap_img.save(f"visualization_images/heatmap_unit_{u}.png")

num_units = acts.shape[1]

pbar.descnext('rq')
def compute_samples(batch, *args):
    image_batch = batch.cuda()
    _ = model(image_batch)
    acts = model.retained_layer(layername)
    hacts = upfn(acts)
    return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])
rq = tally.tally_quantile(compute_samples, dataset,
                          sample_size=sample_size,
                          r=8192,
                          num_workers=100,
                          pin_memory=True,
                          cachefile=resfile('rq.npz'))

pbar.descnext('topk')
def compute_image_max(batch, *args):
    image_batch = batch.cuda()
    _ = model(image_batch)
    acts = model.retained_layer(layername)
    acts = acts.view(acts.shape[0], acts.shape[1], -1)
    acts = acts.max(2)[0]
    return acts
topk = tally.tally_topk(compute_image_max, dataset, sample_size=sample_size,
        batch_size=50, num_workers=30, pin_memory=True,
        cachefile=resfile('topk.npz'))

# single image visualization
print(topk.result()[1][10][6], dataset.images[topk.result()[1][10][6]])
image_number = topk.result()[1][10][4].item()
unit_number = 10
iv = imgviz.ImageVisualizer((224, 224), source=dataset, quantiles=rq,
        level=rq.quantiles(percent_level))
batch = torch.cat([dataset[i][0][None,...] for i in [image_number]])
truth = [classlabels[dataset[i][1]] for i in [image_number]]
preds = model(batch.cuda()).max(1)[1]
imgs = [renormalize.as_image(t, source=dataset) for t in batch]
prednames = [classlabels[p.item()] for p in preds]
acts = model.retained_layer(layername)
# 保存图像到文件
for i, (img, pred, gt) in enumerate(zip(imgs, prednames, truth)):
    img.save(f"later_image_{i}_pred_{pred}_true_{gt}.png")

# 保存单个图像的可视化
masked_img = iv.masked_image(batch[0], acts, (0, unit_number))
masked_img.save(f"masked_image_{image_number}_unit_{unit_number}.png")

pbar.descnext('unit_images')

iv = imgviz.ImageVisualizer((100, 100), source=dataset, quantiles=rq,
        level=rq.quantiles(percent_level))
def compute_acts(image_batch):
    image_batch = image_batch.cuda()
    _ = model(image_batch)
    acts_batch = model.retained_layer(layername)
    return acts_batch
unit_images = iv.masked_images_for_topk(
        compute_acts, dataset, topk, k=5, num_workers=30, pin_memory=True,
        cachefile=resfile('top5images.npz'))


# 保存图像到文件
for u in [10, 20, 30, 40, 19, 190]:
    print('unit %d' % u)
    img = unit_images[u]
    img.save(f"unit_images/unit_{u}.png")

