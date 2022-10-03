import torch
import torch.nn as nn
import numpy as np
from depth.models.builder import LOSSES
import torch.nn.functional as F


###########
# EDGE-GUIDED SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w
# return:
# inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B
###########
def ind2sub(idx, cols):
    r = idx / cols
    c = idx - r * cols
    return r, c

def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx


def edgeGuidedSampling(inputs, targets, edges_img, thetas_img, masks, h, w):
    # find edges
    edges_max = edges_img.max()
    edges_min = edges_img.min()
    edges_mask = edges_img.ge(edges_max*0.1)
    edges_loc = edges_mask.nonzero()

    thetas_edge = torch.masked_select(thetas_img, edges_mask)
    minlen = thetas_edge.size()[0]

    # find anchor points (i.e, edge points)
    sample_num = minlen
    index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long).cuda()
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
    ## compute the coordinates of 4-points,  distances are from [2, 30]
    distance_matrix = torch.randint(3, 20, (4,sample_num)).cuda()
    pos_or_neg = torch.ones(4,sample_num).cuda()
    pos_or_neg[:2,:] = -pos_or_neg[:2,:]
    distance_matrix = distance_matrix.float() * pos_or_neg
    col = col_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.cos(theta_anchors).unsqueeze(0)).long()
    row = row_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.double() * torch.sin(theta_anchors).unsqueeze(0)).long()

    # constrain 0=<c<=w, 0<=r<=h
    # Note: index should minus 1
    col[col<0] = 0
    col[col>w-1] = w-1
    row[row<0] = 0
    row[row>h-1] = h-1

    # a-b, b-c, c-d
    a = sub2ind(row[0,:], col[0,:], w)
    b = sub2ind(row[1,:], col[1,:], w)
    c = sub2ind(row[2,:], col[2,:], w)
    d = sub2ind(row[3,:], col[3,:], w)
    A = torch.cat((a,b,c), 0)
    B = torch.cat((b,c,d), 0)

    inputs_A = inputs[:, A]
    inputs_B = inputs[:, B]
    targets_A = targets[:, A]
    targets_B = targets[:, B]
    masks_A = torch.gather(masks, 0, A.long())
    masks_B = torch.gather(masks, 0, B.long())
    return inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num, row, col

def randomSamplingNormal(inputs, targets, masks, sample_num):

    # find A-B point pairs from predictions
    num_effect_pixels = torch.sum(masks)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels).cuda()
    valid_inputs = inputs[:, masks]
    valid_targes = targets[:, masks]
    inputs_A = valid_inputs[:, shuffle_effect_pixels[0:sample_num*2:2]]
    inputs_B = valid_inputs[:, shuffle_effect_pixels[1:sample_num*2:2]]
    # find corresponding pairs from GT
    targets_A = valid_targes[:, shuffle_effect_pixels[0:sample_num*2:2]]
    targets_B = valid_targes[:, shuffle_effect_pixels[1:sample_num*2:2]]
    if inputs_A.shape[1] != inputs_B.shape[1]:
        num_min = min(targets_A.shape[1], targets_B.shape[1])
        inputs_A = inputs_A[:, :num_min]
        inputs_B = inputs_B[:, :num_min]
        targets_A = targets_A[:, :num_min]
        targets_B = targets_B[:, :num_min]
    return inputs_A, inputs_B, targets_A, targets_B

def init_image_coor(height, width):
    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x[np.newaxis, :, :]
    x = x.astype(np.float32)
    x = torch.from_numpy(x.copy()).cuda()
    u_u0 = x - width/2.0

    y_col = np.arange(0, height)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y[np.newaxis, :, :]
    y = y.astype(np.float32)
    y = torch.from_numpy(y.copy()).cuda()
    v_v0 = y - height/2.0
    return u_u0, v_v0


def depth_to_xyz(depth, focal_length):
    b, c, h, w = depth.shape
    u_u0, v_v0 = init_image_coor(h, w)
    x = u_u0 * depth / focal_length
    y = v_v0 * depth / focal_length
    z = depth
    pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
    return pw


def get_surface_normal(xyz, patch_size=5):
    # xyz: [1, h, w, 3]
    x, y, z = torch.unbind(xyz, dim=3)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 0)
    z = torch.unsqueeze(z, 0)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    patch_weight = torch.ones((1, 1, patch_size, patch_size), requires_grad=False).cuda()
    xx_patch = nn.functional.conv2d(xx, weight=patch_weight, padding=int(patch_size / 2))
    yy_patch = nn.functional.conv2d(yy, weight=patch_weight, padding=int(patch_size / 2))
    zz_patch = nn.functional.conv2d(zz, weight=patch_weight, padding=int(patch_size / 2))
    xy_patch = nn.functional.conv2d(xy, weight=patch_weight, padding=int(patch_size / 2))
    xz_patch = nn.functional.conv2d(xz, weight=patch_weight, padding=int(patch_size / 2))
    yz_patch = nn.functional.conv2d(yz, weight=patch_weight, padding=int(patch_size / 2))
    ATA = torch.stack([xx_patch, xy_patch, xz_patch, xy_patch, yy_patch, yz_patch, xz_patch, yz_patch, zz_patch],
                      dim=4)
    ATA = torch.squeeze(ATA)
    ATA = torch.reshape(ATA, (ATA.size(0), ATA.size(1), 3, 3))
    eps_identity = 1e-6 * torch.eye(3, device=ATA.device, dtype=ATA.dtype)[None, None, :, :].repeat([ATA.size(0), ATA.size(1), 1, 1])
    ATA = ATA + eps_identity
    x_patch = nn.functional.conv2d(x, weight=patch_weight, padding=int(patch_size / 2))
    y_patch = nn.functional.conv2d(y, weight=patch_weight, padding=int(patch_size / 2))
    z_patch = nn.functional.conv2d(z, weight=patch_weight, padding=int(patch_size / 2))
    AT1 = torch.stack([x_patch, y_patch, z_patch], dim=4)
    AT1 = torch.squeeze(AT1)
    AT1 = torch.unsqueeze(AT1, 3)

    patch_num = 4
    patch_x = int(AT1.size(1) / patch_num)
    patch_y = int(AT1.size(0) / patch_num)
    n_img = torch.randn(AT1.shape).cuda()
    overlap = patch_size // 2 + 1
    for x in range(int(patch_num)):
        for y in range(int(patch_num)):
            left_flg = 0 if x == 0 else 1
            right_flg = 0 if x == patch_num -1 else 1
            top_flg = 0 if y == 0 else 1
            btm_flg = 0 if y == patch_num - 1 else 1
            at1 = AT1[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                  x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
            ata = ATA[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                  x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
            n_img_tmp, _ = torch.solve(at1, ata)

            n_img_tmp_select = n_img_tmp[top_flg * overlap:patch_y + top_flg * overlap, left_flg * overlap:patch_x + left_flg * overlap, :, :]
            n_img[y * patch_y:y * patch_y + patch_y, x * patch_x:x * patch_x + patch_x, :, :] = n_img_tmp_select

    n_img_L2 = torch.sqrt(torch.sum(n_img ** 2, dim=2, keepdim=True))
    n_img_norm = n_img / n_img_L2

    # re-orient normals consistently
    orient_mask = torch.sum(torch.squeeze(n_img_norm) * torch.squeeze(xyz), dim=2) > 0
    n_img_norm[orient_mask] *= -1
    return n_img_norm

def get_surface_normalv2(xyz, patch_size=5):
    """
    xyz: xyz coordinates
    patch: [p1, p2, p3,
            p4, p5, p6,
            p7, p8, p9]
    surface_normal = [(p9-p1) x (p3-p7)] + [(p6-p4) - (p8-p2)]
    return: normal [h, w, 3, b]
    """
    b, h, w, c = xyz.shape
    half_patch = patch_size // 2
    xyz_pad = torch.zeros((b, h + patch_size - 1, w + patch_size - 1, c), dtype=xyz.dtype, device=xyz.device)
    xyz_pad[:, half_patch:-half_patch, half_patch:-half_patch, :] = xyz

    # xyz_left_top = xyz_pad[:, :h, :w, :]  # p1
    # xyz_right_bottom = xyz_pad[:, -h:, -w:, :]# p9
    # xyz_left_bottom = xyz_pad[:, -h:, :w, :]   # p7
    # xyz_right_top = xyz_pad[:, :h, -w:, :]  # p3
    # xyz_cross1 = xyz_left_top - xyz_right_bottom  # p1p9
    # xyz_cross2 = xyz_left_bottom - xyz_right_top  # p7p3

    xyz_left = xyz_pad[:, half_patch:half_patch + h, :w, :]  # p4
    xyz_right = xyz_pad[:, half_patch:half_patch + h, -w:, :]  # p6
    xyz_top = xyz_pad[:, :h, half_patch:half_patch + w, :]  # p2
    xyz_bottom = xyz_pad[:, -h:, half_patch:half_patch + w, :]  # p8
    xyz_horizon = xyz_left - xyz_right  # p4p6
    xyz_vertical = xyz_top - xyz_bottom  # p2p8

    xyz_left_in = xyz_pad[:, half_patch:half_patch + h, 1:w+1, :]  # p4
    xyz_right_in = xyz_pad[:, half_patch:half_patch + h, patch_size-1:patch_size-1+w, :]  # p6
    xyz_top_in = xyz_pad[:, 1:h+1, half_patch:half_patch + w, :]  # p2
    xyz_bottom_in = xyz_pad[:, patch_size-1:patch_size-1+h, half_patch:half_patch + w, :]  # p8
    xyz_horizon_in = xyz_left_in - xyz_right_in  # p4p6
    xyz_vertical_in = xyz_top_in - xyz_bottom_in  # p2p8

    n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
    n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

    # re-orient normals consistently
    orient_mask = torch.sum(n_img_1 * xyz, dim=3) > 0
    n_img_1[orient_mask] *= -1
    orient_mask = torch.sum(n_img_2 * xyz, dim=3) > 0
    n_img_2[orient_mask] *= -1

    n_img1_L2 = torch.sqrt(torch.sum(n_img_1 ** 2, dim=3, keepdim=True))
    n_img1_norm = n_img_1 / (n_img1_L2 + 1e-8)

    n_img2_L2 = torch.sqrt(torch.sum(n_img_2 ** 2, dim=3, keepdim=True))
    n_img2_norm = n_img_2 / (n_img2_L2 + 1e-8)

    # average 2 norms
    n_img_aver = n_img1_norm + n_img2_norm
    n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver ** 2, dim=3, keepdim=True))
    n_img_aver_norm = n_img_aver / (n_img_aver_L2 + 1e-8)
    # re-orient normals consistently
    orient_mask = torch.sum(n_img_aver_norm * xyz, dim=3) > 0
    n_img_aver_norm[orient_mask] *= -1
    n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  # [h, w, c, b]

    # a = torch.sum(n_img1_norm_out*n_img2_norm_out, dim=2).cpu().numpy().squeeze()
    # plt.imshow(np.abs(a), cmap='rainbow')
    # plt.show()
    return n_img_aver_norm_out#n_img1_norm.permute((1, 2, 3, 0))

def surface_normal_from_depth(depth, focal_length, valid_mask=None):
    # para depth: depth map, [b, c, h, w]
    b, c, h, w = depth.shape
    # focal_length = focal_length[:, None, None, None]
    depth_filter = nn.functional.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
    depth_filter = nn.functional.avg_pool2d(depth_filter, kernel_size=3, stride=1, padding=1)
    xyz = depth_to_xyz(depth_filter, focal_length)
    sn_batch = []
    for i in range(b):
        xyz_i = xyz[i, :][None, :, :, :]
        normal = get_surface_normalv2(xyz_i)
        sn_batch.append(normal)
    sn_batch = torch.cat(sn_batch, dim=3).permute((3, 2, 0, 1))  # [b, c, h, w]
    mask_invalid = (~valid_mask).repeat(1, 3, 1, 1)
    sn_batch[mask_invalid] = 0.0

    return sn_batch


def vis_normal(normal):
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    @para normal: surface normal, [h, w, 3], numpy.array
    """
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    return normal_vis


def vis_normal2(normals):
    '''
    Montage of normal maps. Vectors are unit length and backfaces thresholded.
    '''
    x = normals[:, :, 0] # horizontal; pos right
    y = normals[:, :, 1] # depth; pos far
    z = normals[:, :, 2] # vertical; pos up
    backfacing = (z > 0)
    norm = np.sqrt(np.sum(normals**2, axis=2))
    zero = (norm < 1e-5)
    x += 1.0; x *= 0.5
    y += 1.0; y *= 0.5
    z = np.abs(z)
    x[zero] = 0.0
    y[zero] = 0.0
    z[zero] = 0.0
    normals[:, :, 0] = x  # horizontal; pos right
    normals[:, :, 1] = y  # depth; pos far
    normals[:, :, 2] = z # vertical; pos up
    return normals

@LOSSES.register_module()
class VNL_Loss(nn.Module):
    def __init__(self, 
        fx,
        point_pairs=10000, 
        cos_theta1=0.3, 
        cos_theta2=0.95, 
        cos_theta3=0.5, 
        cos_theta4=0.86, 
        mask_value=1e-3, 
        loss_weight=1.0, 
        max_depth=10.0):
        super(VNL_Loss, self).__init__()
        self.fx = fx
        self.point_pairs = point_pairs # number of point pairs
        self.mask_value = mask_value
        self.max_threshold = max_depth
        self.loss_weight = loss_weight
        self.cos_theta1 = cos_theta1  # 75 degree
        self.cos_theta2 = cos_theta2  # 10 degree
        self.cos_theta3 = cos_theta3  # 60 degree
        self.cos_theta4 = cos_theta4  # 30 degree
        self.kernel = torch.tensor(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32), requires_grad=False)[None, None, :, :].cuda()

    def scale_shift_pred_depth(self, pred, gt):
        b, c, h, w = pred.shape
        mask = (gt > self.mask_value) & (gt < self.max_threshold)  # [b, c, h, w]
        EPS = 1e-6 * torch.eye(2, dtype=pred.dtype, device=pred.device)
        scale_shift_batch = []
        ones_img = torch.ones((1, h, w), dtype=pred.dtype, device=pred.device)
        for i in range(b):
            mask_i = mask[i, ...]
            pred_valid_i = pred[i, ...][mask_i]
            ones_i = ones_img[mask_i]
            pred_valid_ones_i = torch.stack((pred_valid_i, ones_i), dim=0)  # [c+1, n]
            A_i = torch.matmul(pred_valid_ones_i, pred_valid_ones_i.permute(1, 0))  # [2, 2]
            A_inverse = torch.inverse(A_i + EPS)

            gt_i = gt[i, ...][mask_i]
            B_i = torch.matmul(pred_valid_ones_i, gt_i)[:, None]  # [2, 1]
            scale_shift_i = torch.matmul(A_inverse, B_i)  # [2, 1]
            scale_shift_batch.append(scale_shift_i)
        scale_shift_batch = torch.stack(scale_shift_batch, dim=0)  # [b, 2, 1]
        ones = torch.ones_like(pred)
        pred_ones = torch.cat((pred, ones), dim=1)  # [b, 2, h, w]
        pred_scale_shift = torch.matmul(pred_ones.permute(0, 2, 3, 1).reshape(b, h * w, 2), scale_shift_batch)  # [b, h*w, 1]
        pred_scale_shift = pred_scale_shift.permute(0, 2, 1).reshape((b, c, h, w))
        return pred_scale_shift

    def getEdge(self, images):
        n,c,h,w = images.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda().view((1,1,3,3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:,0,:,:].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:,0,:,:].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)
        return edges, thetas

    def getNormalEdge(self, normals):
        n,c,h,w = normals.size()
        a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().view((1,1,3,3)).repeat(3, 1, 1, 1)
        b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda().view((1,1,3,3)).repeat(3, 1, 1, 1)
        gradient_x = torch.abs(F.conv2d(normals, a, groups=c))
        gradient_y = torch.abs(F.conv2d(normals, b, groups=c))
        gradient_x = gradient_x.mean(dim=1, keepdim=True)
        gradient_y = gradient_y.mean(dim=1, keepdim=True)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)
        return edges, thetas

    def VNL_loss(self, gt_depth, pred_depth, images):
        """
        inputs and targets: surface normal image
        images: rgb images
        """
        focal_length = self.fx
        masks = gt_depth > self.mask_value
        #pred_depth_ss = self.scale_shift_pred_depth(pred_depth, gt_depth)
        inputs = surface_normal_from_depth(pred_depth, focal_length, valid_mask=masks)
        targets = surface_normal_from_depth(gt_depth, focal_length, valid_mask=masks)
        # find edges from RGB
        edges_img, thetas_img = self.getEdge(images)
        # find edges from normals
        edges_normal, thetas_normal = self.getNormalEdge(targets)
        mask_img_border = torch.ones_like(edges_normal)  # normals on the borders
        mask_img_border[:, :, 5:-5, 5:-5] = 0
        edges_normal[mask_img_border.bool()] = 0
        # find edges from depth
        edges_depth, _ = self.getEdge(gt_depth)
        edges_depth_mask = edges_depth.ge(edges_depth.max() * 0.1)
        edges_mask_dilate = torch.clamp(torch.nn.functional.conv2d(edges_depth_mask.float(), self.kernel, padding=(1, 1)), 0,
                                       1).bool()
        edges_normal[edges_mask_dilate] = 0
        edges_img[edges_mask_dilate] = 0
        #=============================
        n,c,h,w = targets.size()

        inputs = inputs.contiguous().view(n, c, -1).double()
        targets = targets.contiguous().view(n, c, -1).double()
        masks = masks.contiguous().view(n, -1)
        edges_img = edges_img.contiguous().view(n, -1).double()
        thetas_img = thetas_img.contiguous().view(n, -1).double()
        edges_normal = edges_normal.view(n, -1).double()
        thetas_normal = thetas_normal.view(n, -1).double()

        # initialization
        loss = torch.DoubleTensor([0.0]).cuda()

        for i in range(n):
            # Edge-Guided sampling
            inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num, row_img, col_img = edgeGuidedSampling(inputs[i,:], targets[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w)
            normal_inputs_A, normal_inputs_B, normal_targets_A, normal_targets_B, normal_masks_A, normal_masks_B, normal_sample_num, row_normal, col_normal = edgeGuidedSampling(inputs[i,:], targets[i, :], edges_normal[i], thetas_normal[i], masks[i, :], h, w)


            # Combine EGS + EGNS
            inputs_A = torch.cat((inputs_A, normal_inputs_A), 1)
            inputs_B = torch.cat((inputs_B, normal_inputs_B), 1)
            targets_A = torch.cat((targets_A, normal_targets_A), 1)
            targets_B = torch.cat((targets_B, normal_targets_B), 1)
            masks_A = torch.cat((masks_A, normal_masks_A), 0)
            masks_B = torch.cat((masks_B, normal_masks_B), 0)

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A & masks_B

            #GT ordinal relationship
            target_cos = torch.abs(torch.sum(targets_A * targets_B, dim=0))
            input_cos = torch.abs(torch.sum(inputs_A * inputs_B, dim=0))
            # ranking regression
            #loss += torch.mean(torch.abs(target_cos[consistency_mask] - input_cos[consistency_mask]))

            # Ranking for samples
            mask_cos75 = target_cos < self.cos_theta1
            mask_cos10 = target_cos > self.cos_theta2
            # Regression for samples
            loss += torch.sum(torch.abs(target_cos[mask_cos75 & consistency_mask] - input_cos[mask_cos75 & consistency_mask])) / (torch.sum(mask_cos75 & consistency_mask)+1e-8)
            loss += torch.sum(torch.abs(target_cos[mask_cos10 & consistency_mask] - input_cos[mask_cos10 & consistency_mask])) / (torch.sum(mask_cos10 & consistency_mask)+1e-8)

            # Random Sampling regression
            random_sample_num = torch.sum(mask_cos10 & consistency_mask) + torch.sum(torch.sum(mask_cos75 & consistency_mask))
            random_inputs_A, random_inputs_B, random_targets_A, random_targets_B = randomSamplingNormal(inputs[i,:], targets[i, :], masks[i, :], random_sample_num)
            #GT ordinal relationship
            random_target_cos = torch.abs(torch.sum(random_targets_A * random_targets_B, dim=0))
            random_input_cos = torch.abs(torch.sum(random_inputs_A * random_inputs_B, dim=0))
            loss += torch.sum(torch.abs(random_target_cos - random_input_cos)) / (random_target_cos.shape[0] + 1e-8)

        if loss[0] != 0:
            return loss[0].float() / n
        else:
            return pred_depth.sum() * 0.0

    def forward(self,
                depth_pred,
                depth_gt,
                images,
                **kwargs):
        """Forward function."""

        loss_depth = self.loss_weight * self.VNL_loss(depth_gt, depth_pred, images)


        return loss_depth
