#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from os import makedirs
import torchvision
from utils.loss_utils import l1_loss, ssim, compute_depth_loss
from utils.vis_utils import vis_cameras
from utils import depth_utils
from torchmetrics.functional.regression import pearson_corrcoef
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, visualize_depth, check_socket_open
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.cameras import Lie, evaluate_camera_alignment, prealign_cameras, align_cameras
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from lpipsPyTorch import lpips
from load_data_event import load_data
from utils import event_utils
from utils.math_utils import rgb2brightlog
import numpy as np
from loss import imgloss
from utils import img_utils
import visdom
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0

    events = load_data(args.source_path, args)
    mse_loss = imgloss.MSELoss()
    rgb2gray = img_utils.RGB2Gray()



    if not opt.deblur:
        opt.blur_sample_num = 1

    # init tensorboard
    tb_writer = prepare_output_and_logger(dataset,args)

    # init visdom
    is_open = check_socket_open(dataset.visdom_server, dataset.visdom_port)
    retry = None
#    while not is_open:
#        retry = input(
#            "visdom port ({}:{}) not open, retry? (y/n) ".format(dataset.visdom_server, #dataset.visdom_port))
#        if retry not in ["y", "n"]:
#            continue
#        if retry == "y":
#            is_open = check_socket_open(
#                dataset.visdom_server, dataset.visdom_port)
#        else:
#            break
    vis = visdom.Visdom(
        server=dataset.visdom_server, port=dataset.visdom_port)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    #print(checkpoint)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        #print("model_params",model_params)
        print(dir(gaussians), vars(gaussians))

    blur_blend_embedding = torch.nn.Embedding(
        len(scene.getTrainCameras()), opt.blur_sample_num).cuda()
    blur_blend_embedding.weight = torch.nn.Parameter(torch.ones(
        len(scene.getTrainCameras()), opt.blur_sample_num).cuda())
    optimizer = torch.optim.Adam([
        {'params': blur_blend_embedding.parameters(),
         'lr': 1e-3, "name": "blur blend parameters"},
    ], lr=0.0, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(1e-6/1e-3)**(1./opt.iterations))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations),
                        desc="Training progress")
    first_iter += 1

    # train_cameras = scene.getTrainCameras()
    # test_cameras = scene.getTestCameras()
    # train_pose = torch.stack([cam.get_pose() for cam in train_cameras])
    # train_pose_GT = torch.stack([cam.pose_gt for cam in train_cameras])
    # test_pose_gt = torch.stack([cam.pose_gt for cam in test_cameras])
    # aligned_train_pose, sim3 = prealign_cameras(train_pose, train_pose_GT)
    # aligned_test_pose = align_cameras(sim3, test_pose_gt)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(
                        net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % opt.sh_up_degree_interval == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()

        #viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack[args.index]
        #print("uid=",viewpoint_cam.uid, viewpoint_cam.image_name)
        blur_weight = blur_blend_embedding(
            torch.tensor(viewpoint_cam.uid).cuda())
        blur_weight /= torch.sum(blur_weight)
        #print("blur_weight.shape",blur_weight.shape)
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand(
            (3), device="cuda") if opt.random_background else background

        image = 0
        depth = 0
        radii = None
        viewspace_point_tensors = []
        viewspace_point_tensor_data = 0
        visibility_filter = None
        per_rgb_loss = 0

        # Loss
        if opt.ground_truth:
            gt_image = viewpoint_cam.test_image.cuda()
        else:
            gt_image = viewpoint_cam.original_image.cuda()
        predict_depth = viewpoint_cam.predict_depth.cuda()

        if not opt.non_uniform:
            blur_weight = 1.0/opt.blur_sample_num

        for idx in range(opt.blur_sample_num):
            alpha = idx / (max(1, opt.blur_sample_num-1))
            render_pkg = render(viewpoint_cam, gaussians,
                                pipe, bg, interp_alpha=alpha)
            image_, depth_, viewspace_point_tensor_, visibility_filter_, radii_ = render_pkg["render"], render_pkg[
                "depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            if iteration in saving_iterations:
                render_path = os.path.join("output", args.source_path.split("/")[-1]+"_"+str(args.index)+"_rgb_"+str(args.rgb_coeff)+"_event_"+str(args.event_coeff_real),"train","train_{}".format(iteration),"rgb")
                makedirs(render_path, exist_ok=True)
                torchvision.utils.save_image(image_, os.path.join(
                    render_path, '{0:05d}'.format(idx) + "_" + str(iteration)+ ".jpg"))
            image += image_*blur_weight
            depth += depth_*blur_weight
            radii = radii_ if radii is None else torch.max(radii_, radii)
            per_rgb_loss += l1_loss(image_, gt_image)*blur_weight
            visibility_filter = visibility_filter_ if visibility_filter is None else torch.logical_or(
                visibility_filter, visibility_filter_)
            viewspace_point_tensors.append(viewspace_point_tensor_)
            viewspace_point_tensor_data += viewspace_point_tensor_ * \
                blur_weight



        if args.event_time_window:
            window_t = args.accumulate_time_length
            if args.random_sampling_window:
                low_t = np.random.rand(1) * (1 - window_t)
                upper_t = low_t + window_t
            else:
                low_t = np.random.randint((1 - window_t) // window_t) * window_t
                upper_t = np.min((low_t + window_t, 1.0))
            idx_a = low_t <= events["ts"]
            idx_b = events["ts"] <= upper_t
            idx = idx_a * idx_b
            indices = np.where(idx)
            # get element in time window
            pol_window = events['pol'][indices]
            x_window = events['x'][indices]
            y_window = events['y'][indices]
            ts_window = events['ts'][indices]
        else:
            num = len(events["pol"])
            N_window = round(num * args.accumulate_time_length)
            if args.random_sampling_window:
                window_low_bound = np.random.randint(num - N_window)
                window_up_bound = int(window_low_bound + N_window)
            else:
                window_low_bound = np.random.randint((num - N_window) // N_window) * N_window
                window_up_bound = int(window_low_bound + N_window)
            pol_window = events['pol'][window_low_bound:window_up_bound]
            x_window = events['x'][window_low_bound:window_up_bound]
            y_window = events['y'][window_low_bound:window_up_bound]
            ts_window = events['ts'][window_low_bound:window_up_bound]
        Ne=10000
        out = np.zeros((args.event_height, args.event_width))
        #print("x_window.shape",x_window[:Ne,].shape,low_t,ts_window[Ne],upper_t)
        if x_window.shape[0]<10001:
            events_accu = event_utils.accumulate_events_on_gpu(out, x_window, y_window, pol_window)
        else:
            events_accu = event_utils.accumulate_events_on_gpu(out, x_window[:Ne,], y_window[:Ne,], pol_window[:Ne,])
            upper_t = [ts_window[Ne]]
        target_s = events_accu
        # event_utils.accumulate_events(out, x_window, y_window, pol_window)
        # events_accu = torch.tensor(out)

        # timestamps of event windows begin and end
        if args.event_time_window:
            events_ts = np.stack((low_t, upper_t)).reshape(2)
        else:
            events_ts = ts_window[np.array([0, int(N_window) - 1])]

        #print(events_ts)
        events_render=[]
        count = 0
        #print(events_ts)
        for sample in events_ts:
            count=count+1
            render_pkg = render(viewpoint_cam, gaussians,
                                pipe, bg, interp_alpha=sample)
            #events_render.append(render_pkg["render"])
            color_event = render_pkg["render"]
            if iteration in saving_iterations:
                render_path = os.path.join("output", args.source_path.split("/")[-1]+"_"+str(args.index)+"_rgb_"+str(args.rgb_coeff)+"_event_"+str(args.event_coeff_real),"train","train_{}".format(iteration),"event")
                makedirs(render_path, exist_ok=True)
                torchvision.utils.save_image(color_event, os.path.join(
                    render_path, '{0:05d}'.format(count) + "_" + str(iteration)+ ".jpg"))
            gray_event = 0.299*color_event[0,:,:]+0.587*color_event[1,:,:]+0.114*color_event[2,:,:]
            events_render.append(rgb2brightlog(gray_event, args.dataset))
        event_render_accumulate = events_render[1]-events_render[0]
        render_brightness_diff = event_render_accumulate
        render_norm = render_brightness_diff / (
                        torch.linalg.norm(render_brightness_diff, dim=0, keepdim=True) + 1e-9
                    )
        target_s_norm = target_s / (
                        torch.linalg.norm(target_s, dim=0, keepdim=True) + 1e-9
                    )
        #print("render_norm.shape, target_s_norm.shape",torch.min(target_s_norm),torch.max(target_s_norm),torch.min(render_norm),torch.max(render_norm))
        event_loss = mse_loss(render_norm, target_s_norm)
        if iteration in saving_iterations:
            #print("event_gt_accu min and max", torch.min(events_accu), torch.max(events_accu))
            #print("event_render_accu min and max", torch.min(event_render_accumulate), torch.max(event_render_accumulate))
            #print("rgb_gt min and max", torch.min(gt_image), torch.max(gt_image))
            torchvision.utils.save_image(events_accu, os.path.join(
                    render_path + "_gt" + "_" + str(iteration)+ ".jpg"))
            torchvision.utils.save_image(event_render_accumulate, os.path.join(
                    render_path + "_render" + "_" + str(iteration)+ ".jpg"))
        #print(event_render_accumulate.shape, events_accu.shape)

        image_loss = l1_loss(image, gt_image)
        #loss = event_loss
        #if iteration<1001:
         #   event_coeff_real=0.0
          #  rgb_coeff=args.rgb_coeff
        #else:
         #   event_coeff_real=args.event_coeff_real
         #   rgb_coeff=args.rgb_coeff
        #print("iteration, event_coeff_real,rgb_coeff",iteration, event_coeff_real,rgb_coeff)
        loss =args.event_coeff_real*event_loss + args.rgb_coeff*((1.0 - opt.lambda_dssim) * image_loss + \
            opt.lambda_dssim * (1.0 - ssim(image, gt_image))) \

        if opt.depth_reg:
            loss += opt.depth_weight * min(
                (1 - pearson_corrcoef(- predict_depth, depth)),
                (1 - pearson_corrcoef(1 / (predict_depth + 200.), depth))
            )

        loss.backward()

        viewspace_point_tensor = viewspace_point_tensor_data.clone().detach().requires_grad_(True)
        viewspace_point_tensor.grad = None
        for viewspace_point_tensor_ in viewspace_point_tensors:
            if viewspace_point_tensor.grad is None:
                viewspace_point_tensor.grad = viewspace_point_tensor_.grad
            else:
                viewspace_point_tensor.grad = torch.max(
                    viewspace_point_tensor.grad, viewspace_point_tensor_.grad)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:
                print(f"blur_weight={blur_weight}")
            # Log and save
            training_report(tb_writer, vis, iteration, image_loss, loss, l1_loss, iter_start.elapsed_time(
                iter_end), testing_iterations, scene, pipe, background, opt,args)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if opt.deblur:
                    viewpoint_cam.update(iteration)
                #     viewpoint_cam.pose_optimizer.step()
                #     viewpoint_cam.pose_scheduler.step()
                #     viewpoint_cam.pose_optimizer.zero_grad(set_to_none=True)

                # if opt.depth_reg:
                #     viewpoint_cam.depth_optimizer.step()
                #     viewpoint_cam.depth_scheduler.step()
                #     viewpoint_cam.depth_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # if iteration % 2 == 0:
            #     train_cameras = scene.getTrainCameras()
            #     test_cameras = scene.getTestCameras()
            #     train_pose = torch.stack([cam.get_pose()
            #                              for cam in train_cameras])
            #     train_pose_GT = torch.stack(
            #         [cam.pose_gt for cam in train_cameras])
            #     test_pose_gt = torch.stack(
            #         [cam.pose_gt for cam in test_cameras])
            #     aligned_train_pose, sim3 = prealign_cameras(
            #         train_pose, train_pose_GT)

                # vis_cameras(vis, step=iteration, poses=[
                #             aligned_train_pose, train_pose_GT])


def prepare_output_and_logger(args,args_):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", args_.source_path.split("/")[-1]+"_"+str(args_.index)+"_rgb_"+str(args_.rgb_coeff)+"_event_"+str(args_.event_coeff_real))

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, vis, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, pipe, background, opt,args):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss',
                             Ll1.item(), iteration)
        tb_writer.add_scalar(
            'train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        train_pose = torch.stack([cam.get_pose() for cam in train_cameras])
        train_pose_GT = torch.stack([cam.pose_gt for cam in train_cameras])
        #test_pose_gt = torch.stack([cam.pose_gt for cam in test_cameras])
        aligned_train_pose, sim3 = prealign_cameras(train_pose, train_pose_GT)
        #aligned_test_pose = align_cameras(sim3, test_pose_gt)

        vis_cameras(vis, step=iteration, poses=[
                    aligned_train_pose, train_pose_GT])

        # validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
        #                       {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        validation_configs = ({'name': 'test', 'cameras': [scene.getTrainCameras()[args.index]]},{'name': 'train', 'cameras': [scene.getTrainCameras()[args.index]]})

        for config in validation_configs:
            #print("config['cameras']:",config['cameras'])
            if config['cameras'] and len(config['cameras']) > 0:
                rgb_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                depth_test = 0.0
                lpips_test = 0.0
                #print("config[cameras]",len(config['cameras']))
                for idx, viewpoint in enumerate(config['cameras']):
                    for idx_ in range(opt.blur_sample_num):
                        alpha = idx_ / (max(1, opt.blur_sample_num-1))
                        if config['name'] == 'test':
                            continue
                            render_result = render(viewpoint, scene.gaussians,
                                pipe, background, interp_alpha=alpha)
                        else:
                            render_result = render(viewpoint, scene.gaussians,
                                pipe, background, interp_alpha=alpha)
                            #print(f"{idx} gaussian_trans={viewpoint.gaussian_trans}")

                        image = torch.clamp(render_result["render"], 0.0, 1.0)
                        #os.makedirs("test_", exist_ok=True)
                        #torchvision.utils.save_image(image, os.path.join(
                         #   "test_", '{0:05d}'.format(idx_)+ ".jpg"))
                        gt_image = torch.clamp(
                            viewpoint.test_image.cuda(), 0.0, 1.0)
                        ref_image = torch.clamp(
                            viewpoint.original_image.cuda(), 0.0, 1.0)
                        predict_depth = viewpoint.rescale_depth(
                            viewpoint.predict_depth.cuda())
                        vis_predict_depth = visualize_depth(predict_depth)
                        vis_depth = visualize_depth(render_result["depth"])
                        depth_loss = compute_depth_loss(
                            predict_depth, render_result["depth"])
                        if tb_writer:
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(
                                viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/depth".format(
                                viewpoint.image_name), vis_depth[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/predict_depth".format(
                                    viewpoint.image_name), vis_predict_depth[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(
                                    viewpoint.image_name), gt_image[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/reference".format(
                                    viewpoint.image_name), ref_image[None], global_step=iteration)
                        rgb_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                        lpips_test += lpips(image, gt_image,
                                        net_type='vgg').mean().double()
                        depth_test += depth_loss.mean().double()
                print("length:",len(config['cameras']))
                psnr_test /= len(config['cameras'])*opt.blur_sample_num
                ssim_test /= len(config['cameras'])*opt.blur_sample_num
                lpips_test /= len(config['cameras'])*opt.blur_sample_num
                rgb_test /= len(config['cameras'])*opt.blur_sample_num
                depth_test /= len(config['cameras'])*opt.blur_sample_num
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                    iteration, config['name'], rgb_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - image_loss', rgb_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - depth_loss', depth_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    if config['name'] == 'train':
                        cam_error = evaluate_camera_alignment(
                            aligned_train_pose, train_pose_GT)
                        tb_writer.add_scalar(
                            config['name'] + '/loss_viewpoint - R_error', cam_error.R.mean(), iteration)
                        tb_writer.add_scalar(
                            config['name'] + '/loss_viewpoint - t_error', cam_error.t.mean(), iteration)

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(
                'total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+",
                        type=int, default=list(range(1000, 30_000, 1000)))
    #parser.add_argument("--test_iterations", nargs="+",
    #                    type=int, default=list(range(1, 30_000, 10)))
    #parser.add_argument("--save_iterations", nargs="+",
    #                    type=int, default=list(range(1, 30_000, 10)))
    
    parser.add_argument("--save_iterations", nargs="+",
                        type=int, default=list(range(1000, 30_000, 1000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations",
                        nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str)
    #parser.add_argument("--start_checkpoint", type=str, default="output/v1/camera_2_0.1/chkpnt7000.pth")
    parser.add_argument("--index", type=int, default=1,
                        help='the index of the image in the dataset to deblur')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type = str, 
                        default = "E2NeRF_Real", help='the dataset name')
    ## event stream parameters
    parser.add_argument("--event_threshold", type=float, default=0.1,
                        help='threshold set for events spiking')
    parser.add_argument("--event_shift_start", type=float, default=0,
                        help='shift the start timestamp for event stream')
    parser.add_argument("--event_shift_end", type=float, default=0,
                        help='shift the end timestamp for event stream')
    parser.add_argument("--accumulate_time_length", type=float, default=0.1,
                        help='the percentage of the window')
    parser.add_argument("--random_sampling_window", action='store_false',
                        help='whether to use fixed windows or sliding window')
    parser.add_argument("--event_time_window", action='store_false',
                        help='whether to use fixed windows or sliding window')
    parser.add_argument("--event_width", type=int, default=346,
                        help='the width of Event camera image')
    parser.add_argument("--event_height", type=int, default=260,
                        help='the height of Event camera image')
    parser.add_argument("--event_coeff_real", type=float, default=10.0,
                        help='event loss coefficient')
    parser.add_argument("--rgb_coeff", type=float, default=1.0,
                        help='rgb loss coefficient')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.device)

    print(torch.cuda.current_device())
    #print(args)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)


    # All done
    print("\nTraining complete.")
