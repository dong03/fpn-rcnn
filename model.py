from __future__ import absolute_import
from __future__ import division
from util import *

def nogard(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return new_f

class FPN(nn.Module):
    def __init__(self, res):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
        )
        self.res = res
        for name, value in self.res.named_parameters():
            value.requires_grad = False
        self.out_dim = 256
        if opt.simple_model:
            self.layer1 = nn.Sequential(*list(self.res.layer1)[:2])
            self.layer2 = nn.Sequential(*list(self.res.layer2)[:2])
            self.layer3 = nn.Sequential(*list(self.res.layer3)[:2])
            self.layer4 = nn.Sequential(*list(self.res.layer4)[:2])
        else:
            self.layer1 = self.res.layer1
            self.layer2 = self.res.layer2
            self.layer3 = self.res.layer3
            self.layer4 = self.res.layer4
        # Top layer
        self.toplayer = nn.Conv2d(2048, self.out_dim, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, self.out_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, self.out_dim, kernel_size=1, stride=1, padding=0)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # ottom-up
        #pdb.set_trace()
        c1 = self.conv1(x)
        #c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


class RegionProposalNetwork(torch.nn.Module):

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5,1,2],
            anchor_scales=[2,4,8,16,32], feat_stride=[4, 8, 16, 32],
            proposal_creator_params=dict()):
        super(RegionProposalNetwork, self).__init__()

        self.ratios = ratios
        self.anchor_scales = anchor_scales
        self.feat_stride = feat_stride

        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)

        num_anchor_base = 3
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = torch.nn.Conv2d(mid_channels, num_anchor_base*2, 1, 1, 0) #二分类，obj or nobj
        self.loc = torch.nn.Conv2d(mid_channels, num_anchor_base*4, 1, 1, 0) #坐标回归
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, feature_maps, img_size, scale=1.):
        feature_maps_num = len(feature_maps)
        all_anchors = list()
        all_rois = list()
        all_roi_indices = list()
        all_rpn_locs = []
        all_rpn_fg_scores = []
        all_rpn_scores = []


        for i in range(feature_maps_num):
            batch_size, _, hh, ww = feature_maps[i].shape  # x为feature map, n为batch_size,此版本代码为1. _为512, hh, ww即为特征图宽高
            if i == 0:
                anchor_base = generate_anchor_base(anchor_scales=[4], ratios=self.ratios)
            if i == 1:
                anchor_base = generate_anchor_base(anchor_scales=[8], ratios=self.ratios)
            if i == 2:
                anchor_base = generate_anchor_base(anchor_scales=[16], ratios=self.ratios)
            if i == 3:
                anchor_base = generate_anchor_base(anchor_scales=[32], ratios=self.ratios)
            anchor = _enumerate_shifted_anchor(
                np.array(anchor_base), self.feat_stride[i], hh, ww)
            all_anchors.append(anchor)
            num_anchor = anchor.shape[0] // (hh * ww)  #
            h = functional.relu(self.conv1(feature_maps[i]), inplace=True)  #(batch_size, 512, hh, ww)

            rpn_locs = self.loc(h)   #(batch_size, 9*4, hh, ww)
            rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(batch_size,-1, 4)  #转换为(batch_size,hh, ww, 9*4)在转换为(batch_size, hh*ww*9, 4)
            rpn_scores = self.score(h)
            rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()  #转换为(batch_size,hh, ww, 9*2)
            rpn_softmax_scores = functional.softmax(rpn_scores.view(batch_size, hh, ww, num_anchor, 2), dim=4)  #TODO 维度问题
            rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  #得到前景的分类概率

            rpn_fg_scores = rpn_fg_scores.view(batch_size, -1) #得到所有anchor的前景分类概率
            rpn_scores = rpn_scores.view(batch_size, -1, 2)

            all_rpn_locs.append(rpn_locs)
            all_rpn_fg_scores.append(rpn_fg_scores)
            all_rpn_scores.append(rpn_scores)

        all_rpn_locs = torch.cat(all_rpn_locs, 1)
        all_rpn_fg_scores = torch.cat(all_rpn_fg_scores, 1)
        all_rpn_scores = torch.cat(all_rpn_scores, 1)
        all_anchors = np.concatenate(all_anchors, axis=0)

        for i in range(batch_size):
            roi = self.proposal_layer(
                all_rpn_locs[i].cpu().data.numpy(),
                all_rpn_fg_scores[i].cpu().data.numpy(),
                all_anchors, img_size, scale=scale)
            #rpn_locs维度（hh * ww * 9，4），rpn_fg_scores维度为（hh * ww * 9），
            #anchor的维度为（hh * ww * 9，4）， img_size的维度为（3，H，W），H和W是经过数据预处理后的。
            #计算（H / 16）x( W / 16)x9(大概20000)
            #个anchor属于前景的概率，取前12000个并经过NMS得到2000个近似目标框G ^ 的坐标。roi的维度为(2000, 4)

            batch_index = i * np.ones((len(roi),),dtype=np.int32)   #(len(roi), )
            all_rois.append(roi)
            all_roi_indices.append(batch_index)  #记录roi的batch批次

        all_rois = np.concatenate(all_rois,axis=0)  #按列排所有的roi， rois格式（R， 4），R为所有batch的roi数量
        all_roi_indices = np.concatenate(all_roi_indices, axis=0) #按列排所有roi的批次编号，格式同rois

        # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2），
        # rois的维度为（2000,4），roi_indices用不到(因为此代码训练时batch为1)，anchor的维度为（hh*ww*9，4）
        return all_rpn_locs, all_rpn_scores, all_rois, all_roi_indices, all_anchors

class Resnet50RoIHead(torch.nn.Module):
    def __init__(self, n_class, roi_size, feat_stride):
        super(Resnet50RoIHead, self).__init__()
        self.fc6 = torch.nn.Linear(7*7*256, 1024)
        self.fc7 = torch.nn.Linear(1024, 1024)
        self.cls_loc = torch.nn.Linear(1024, n_class * 4)
        self.score = torch.nn.Linear(1024, n_class)
        normal_init(self.fc6, 0, 0.01)
        normal_init(self.fc7, 0, 0.01)
        normal_init(self.cls_loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        #weights_init(self.rcnn_top, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.feat_stride = feat_stride
        self.spatial_scale = [1. / i for i in feat_stride]

    def forward(self, features_maps, rois, roi_indices):
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        roi_level = self._PyramidRoI_Feat(rois)
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]  # yx->xy
        indices_and_rois = xy_indices_and_rois.contiguous()  # 把tensor变成在内存中连续分布的形式

        roi_pool_feats = []
        roi_to_levels = []

        for i, l in enumerate(range(2, 6)):
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero()
            roi_to_levels.append(idx_l)
            #if idx_l.shape[0] == 0:
             #   keep_indices_and_rois = indices_and_rois[idx_l.data]
            #else:
            keep_indices_and_rois = indices_and_rois[idx_l]
            keep_indices_and_rois = keep_indices_and_rois.view(-1, 5)
            roi_pooling = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale[i])
            pool = roi_pooling(features_maps[i], keep_indices_and_rois)   #通过roi_pooling
            roi_pool_feats.append(pool)
        roi_pool_feats = torch.cat(roi_pool_feats, 0)
        roi_to_levels = torch.cat(roi_to_levels, 0)
        roi_to_levels = roi_to_levels.squeeze()
        idx_sorted, order = torch.sort(roi_to_levels)
        roi_pool_feats = roi_pool_feats[order]

        pool = roi_pool_feats.view(roi_pool_feats.size(0), -1)  # batch_size, CHW拉直

        fc6_out = functional.relu(self.fc6(pool))
        fc7_out = functional.relu(self.fc7(fc6_out))
        roi_cls_locs = self.cls_loc(fc7_out)  # （1000->84）每一类坐标回归
        roi_scores = self.score(fc7_out)  # （1000->21） 每一类类别预测
        #all_roi_cls_locs.append(roi_cls_locs)
        #all_roi_scores.append(roi_scores)

        #all_roi_cls_locs = torch.cat(all_roi_cls_locs, 0)
        #all_roi_scores = torch.cat(all_roi_scores, 0)

        return roi_cls_locs, roi_scores

    def _PyramidRoI_Feat(self, rois):
        roi_h = rois[:, 2] - rois[:, 0] + 1
        roi_w = rois[:, 3] - rois[:, 1] + 1
        roi_level = torch.log(torch.sqrt(roi_h*roi_w)/224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        return roi_level

class FasterRCNN(nn.Module):

    def __init__(self,
                 class_num,
                 loc_normalize_mean = (0., 0., 0., 0.),
                 loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        ratios=[0.5, 1, 2]
        anchor_scales=[8]
        feat_stride=[4, 8, 16, 32, 64]
        self.fpn = FPN(
            torchvision.models.resnet152(pretrained= True)
        )

        self.rpn = RegionProposalNetwork(
            256, 256,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=feat_stride,
        )

        self.head = Resnet50RoIHead(
            n_class=class_num + 1,
            roi_size=7,
            feat_stride=[4, 8, 16, 32, 64])
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]   #处理后图片的h和w

        p2, p3, p4, p5 = self.fpn(x)
        features_maps = [p2, p3, p4, p5]
        rcnn_maps = [p2, p3, p4]
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features_maps, img_size, scale)   #rpn网络
        roi_cls_locs, roi_scores = self.head(rcnn_maps, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices


    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):  #将roihead的预测结果利用score_thresh和nms_thresh进行过滤
        bbox = list()
        label = list()
        score = list()
        for l in range(1, self.n_class):  #0为背景
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh

            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            label.append((l-1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


    @nogard
    def predict(self, imgs, sizes = None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = totensor(img[None]).float()   #img增加一维(_, C, H, W)
            scale = img.shape[3] / size[1]   # W' / W, 处理后图像和原图比例
            roi_cls_locs, roi_scores, rois, roi_indices = self(img, scale=scale)

            #batch size为1
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_locs.data
            roi = totensor(rois) / scale

            mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_class)[None]  #(1,84)
            std = torch.Tensor(self.loc_normalize_std).cuda().repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)  #(R, 21 ,4)

            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)  #扩充维度  #(R, 21, 4)
            cls_bbox = loc2bbox(tonumpy(roi).reshape(-1,4),
                                tonumpy(roi_cls_loc).reshape(-1,4))
            cls_bbox = totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)  #(R, 84)
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1]) #裁剪预测bbox不超出原图尺寸

            prob = tonumpy(
                functional.softmax(totensor(roi_score), dim=1))

            raw_cls_bbox = tonumpy(cls_bbox)
            raw_prob = tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)   #将每个batch_size的压在一起

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        lr =opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]

        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
