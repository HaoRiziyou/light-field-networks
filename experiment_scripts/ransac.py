"""
      establish ransac process rec -> test -> rec ->test
      before reconstructing test object, adjust the batch_size :dataloader_params=((64, 1, None),)
      test one instance of plane  data will be:eg 4a559ef6547b685d8aed56c1a220a07d
      typical command: srun --mpi=pmi2 python3 experiment_scripts/ransac.py --data_root=path_to_nmr_dataset --dataset=NMR  --checkpoint=path_to_training_checkpoint --experiment_name=reconstruct_name --test_experiment_name=test_name --gpus=1 --sparsity= the number you want to sparsify(typical 64)

"""


import sys
import os
import numpy as np
import skimage.measure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
import cv2
import torch

import multiclass_dataio
import hdf5_dataio
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Manager

torch.multiprocessing.set_sharing_strategy('file_system')

import models
import training
import configargparse
from collections import defaultdict
import summaries
import config
from torch.utils.data import DataLoader
import util
import shutil
import data_util
from pathlib import Path
import loss_functions

p = configargparse.ArgumentParser()

p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--logging_root', type=str, default=config.logging_root)
p.add_argument('--data_root', type=str, required=True)
p.add_argument('--experiment_name', type=str, default='nmr_rec', required=False)
p.add_argument('--viewlist', type=str, default='./experiment_scripts/viewlists/src_dvr.txt')
p.add_argument('--max_num_instances', type=int, default=None)
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--checkpoint_path', required=True)
p.add_argument('--dataset', type=str, required=True)
p.add_argument('--sparsity', type=int, default=None)

p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--num_epochs', type=int, default=500)
p.add_argument('--epochs_til_ckpt', type=int, default=100)
p.add_argument('--steps_til_summary', type=int, default=100)
p.add_argument('--iters_til_ckpt', type=int, default=50)
p.add_argument('--spec_observation_idcs', type=str, default=None)

p.add_argument('--result_logging_root', type=str, default=config.results_root)
p.add_argument('--test_experiment_name', type=str, required=True)
p.add_argument('--network', type=str, default='relu')
p.add_argument('--conditioning', type=str, default='hyper')
p.add_argument('--save_out_first_n', type=int, default=100, help='Only saves images of first n object instances.')
p.add_argument('--img_sidelength', type=int, default=64, required=False)

opt = p.parse_args()


def convert_image(img, type):
    img = img[0]

    if not 'normal' in type:
        img = util.lin2img(img)[0]
    img = img.cpu().numpy().transpose(1, 2, 0)

    if 'rgb' in type or 'normal' in type:
        img += 1.
        img /= 2.
    elif type == 'depth':
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.
    img = np.clip(img, 0., 255.).astype(np.uint8)
    return img


def get_psnr(p, trgt):
    p = util.lin2img(p.squeeze(), mode='np')
    trgt = util.lin2img(trgt.squeeze(), mode='np')

    p = util.detach_all(p)
    trgt = util.detach_all(trgt)

    p = (p / 2.) + 0.5
    p = np.clip(p, a_min=0., a_max=1.)
    trgt = (trgt / 2.) + 0.5

    ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
    psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

    return psnr, ssim


def multigpu_train(gpu, opt, cache, rand_idcs, sparsity_list):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)

    if opt.spec_observation_idcs is not None:
        specific_observation_idcs = util.parse_comma_separated_integers(opt.spec_observation_idcs)
    else:
        specific_observation_idcs = None

    torch.cuda.set_device(gpu)


    def create_dataloader_callback(sidelength, batch_size, query_sparsity):
        train_dataset = multiclass_dataio.SceneClassDataset(num_context=1, num_trgt=1,
                                                            root_dir=opt.data_root, query_sparsity=opt.sparsity,
                                                            rand_idcs=rand_idcs,
                                                            sparsity_list=sparsity_list,
                                                            img_sidelength=sidelength, vary_context_number=False,
                                                            cache=cache,
                                                            specific_observation_idcs=specific_observation_idcs,
                                                            max_num_instances=opt.max_num_instances,
                                                            dataset_type='test',
                                                            viewlist=opt.viewlist)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=0)
        return train_loader

    num_instances = multiclass_dataio.get_num_instances(opt.data_root, 'train')
    # print("*************num instances:*****",num_instances) 2830
    model = models.LFAutoDecoder(latent_dim=256, num_instances=num_instances, parameterization='plucker').cuda()

    print(f"Loading weights from train {opt.checkpoint_path}...")
    state_dict = torch.load(opt.checkpoint_path)
    state_dict['latent_codes.weight'] = torch.zeros_like(state_dict['latent_codes.weight'])
    model.load_state_dict(state_dict, strict=True)
    
    if opt.gpus > 1:
        sync_model(model)

    # Define the loss
    print(f"until summary")
    summary_fn = summaries.img_summaries
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    loss_fn = loss_functions.LFLoss()

    model_params = [(name, param) for name, param in model.named_parameters() if 'latent_codes' in name]
    optimizers = [torch.optim.Adam(lr=opt.lr, params=[p for _, p in model_params])]


    training.multiscale_training(model=model, dataloader_callback=create_dataloader_callback,
                                                     dataloader_iters=(1000000,), dataloader_params=((64, 1, None),),
                                                     epochs=opt.num_epochs, lr=opt.lr,
                                                     steps_til_summary=opt.steps_til_summary,
                                                     epochs_til_checkpoint=opt.epochs_til_ckpt,
                                                     model_dir=root_path, loss_fn=loss_fn,
                                                     iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
                                                     overwrite=True, optimizers=optimizers,
                                                     rank=gpu, train_function=training.train, gpus=opt.gpus)




def get_rec_model_dir():
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    dataloader_params = ((64, 1, None),)
    for params in dataloader_params:
        model_dir = os.path.join(root_path, '_'.join(map(str, params)))

    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    checkpoint_dir = os.path.join(checkpoint_dir, 'model_final.pth')

    return checkpoint_dir


def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)


def ransac(counter):
    rand_idcs = np.random.choice(opt.img_sidelength ** 2, size=opt.sparsity, replace=False)
    sparsity_list = None
    correct_temp = 0
    manager = Manager()
    shared_dict = manager.dict()

    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt, shared_dict, rand_idcs, sparsity_list))
    else:
        multigpu_train(0, opt, shared_dict, rand_idcs, sparsity_list)

    if sparsity_list is not None:
        print(f"size***: {sparsity_list.size}")
        print(f"sparsity_list***: {sparsity_list}")
    print(f"above is the first train with maybe inliers")

    """
    produce image part
    """
    rec_checkpoint_path = get_rec_model_dir()

    state_dict = torch.load(rec_checkpoint_path)
    num_instances = state_dict['latent_codes.weight'].shape[0]

    if opt.viewlist is not None:
        with open(opt.viewlist, "r") as f:
            tmp = [x.strip().split() for x in f.readlines()]
        viewlist = {
            x[0] + "/" + x[1]: list(map(int, x[2:]))
            for x in tmp
        }
    model = models.LFAutoDecoder(num_instances=num_instances, latent_dim=256, parameterization='plucker',
                                 network=opt.network,
                                 conditioning=opt.conditioning).cuda()
    model.eval()
    print(f"Loading model {counter}")
    model.load_state_dict(state_dict)

    print("Loading dataset")
    if opt.dataset == 'NMR':
        dataset = multiclass_dataio.get_instance_datasets(opt.data_root, sidelen=opt.img_sidelength,
                                                          dataset_type='test',
                                                          max_num_instances=opt.max_num_instances)

    log_dir = Path(opt.logging_root) / opt.test_experiment_name

    if opt.dataset == 'NMR':
        class_psnrs = defaultdict(list)
        class_counter = defaultdict(int)
    
    indicate = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            
            dummy_query = dataset[i][0]
            instance_name = dummy_query['instance_name']
            print(f"instance nmae {instance_name}")
            
            if opt.dataset == 'NMR':
                obj_class = int(dummy_query['class'].cpu().numpy())
                obj_class = multiclass_dataio.class2string_dict[obj_class]

                if class_counter[obj_class] < opt.save_out_first_n:
                    instance_dir = log_dir / f'{obj_class}' / f'{instance_name}'
                    instance_dir.mkdir(exist_ok=True, parents=True)

            for j, query in enumerate(dataset[i]):
                #print(f"dataset{j} : {dataset[i]}")
                model_input = util.assemble_model_input(query, query)
                model_output = model(model_input)

                out_dict = {}
                out_dict['rgb'] = model_output['rgb']
                out_dict['gt_rgb'] = model_input['query']['rgb']

                is_context = False
                if opt.viewlist is not None:
                    key = '/'.join((obj_class, instance_name))
                    if key in viewlist:
                        if j in viewlist[key]:
                            is_context = True
                            indicate = j
                    else:
                        print(f'{key} not in viewlist')
                        continue

                # if opt.dataset != 'NMR' or not is_context:
                psnr, ssim = get_psnr(out_dict['rgb'], out_dict['gt_rgb'])
                if opt.dataset == 'NMR':
                    if not is_context:
                        class_psnrs[obj_class].append((psnr, ssim))

                if opt.dataset == 'NMR' and class_counter[obj_class] < opt.save_out_first_n:
                    for k, v in out_dict.items():
                        img = convert_image(v, k)
                        if k == 'gt_rgb':
                            cv2.imwrite(str(instance_dir / f"{j:06d}_{k}_{counter}_maybeinlier.png"),
                                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        elif k == 'rgb':
                            cv2.imwrite(str(instance_dir / f"{j:06d}_{counter}_maybeinlier.png"),
                                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif i < opt.save_out_first_n:
                    img = convert_image(out_dict['gt_rgb'], 'rgb')
                    cv2.imwrite(str(instance_dir / f"{j:06d}_gt_{counter}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    img = convert_image(out_dict['rgb'], 'rgb')
                    cv2.imwrite(str(instance_dir / f"{j:06d}_{counter}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            if opt.dataset == 'NMR':
                mean_dict = {}
                for k, v in class_psnrs.items():
                    mean = np.mean(np.array(v), axis=0)
                    mean_dict[k] = f"{mean[0]:.3f} {mean[1]:.3f}"
                print(f"mean_dict:{mean_dict}")

                class_counter[obj_class] += 1

    with open(os.path.join(log_dir, f"results_maybeinlier_{counter}.txt"), "w") as out_file:
        if opt.dataset == 'NMR':
            out_file.write(' & '.join(class_psnrs.keys()) + '\n')

            psnrs, ssims = [], []
            for value in class_psnrs.values():
                mean = np.mean(np.array(value), axis=0)
                psnrs.append(mean[0])
                ssims.append(mean[1])

            out_file.write(' & '.join(map(lambda x: f"{x:.3f}", psnrs)) + '\n')
            out_file.write(' & '.join(map(lambda x: f"{x:.3f}", ssims)) + '\n')
    """
        compute the consensus set
    
    """
    
    if opt.sparsity is not None:
        print(f"choose whiche picture input: {instance_dir}/{0:06d}_gt_rgb_{counter}_maybeinlier.png")
        print(f"choose rec image : {instance_dir}/{0:06d}_{counter}_maybeinlier.png")
        gt_img = cv2.imread(str(instance_dir / f"{0:06d}_gt_rgb_{counter}_maybeinlier.png"))
        new_rec_img = cv2.imread(str(instance_dir / f"{0:06d}_{counter}_maybeinlier.png"))    
    
        abs_diff = (cv2.absdiff(new_rec_img[:, :], gt_img[:, :]))
        # set to gray first   
        gray_image = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)

    
        set_threshold = 75
    
        ret, im = cv2.threshold(gray_image, set_threshold, 255, cv2.THRESH_BINARY)
        coords = np.column_stack(np.where(im < set_threshold))
        ids = []
        for x in range(len(coords)):
            position = coords[x][0] * 64 + coords[x][1]
            ids.append(int(position))
        for x in rand_idcs:
            ids.append(x)
        """
        find the also inliers
        """
        sparsity_list = np.array(list(set(ids)))
        print(f"this is the value for the consensus set: {sparsity_list.size}")
        """
         number of inliers > d 
        """
        if sparsity_list.size > 0.8 * 64 * 64:
            print(f"size: {sparsity_list.size}")
            print(f"sparsity_list: {sparsity_list}")
            print(f"here is the second train with also inliers")
    
            """
            second train with also_inliers and maybe_inliers
            """
            if opt.gpus > 1:
                mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt, shared_dict, rand_idcs, sparsity_list))
            else:
                multigpu_train(0, opt, shared_dict, rand_idcs, sparsity_list)
    
            """
                    produce image using test.py part
            """

            rec_checkpoint_path = get_rec_model_dir()
            state_dict = torch.load(rec_checkpoint_path)
            num_instances = state_dict['latent_codes.weight'].shape[0]
    
            if opt.viewlist is not None:
                with open(opt.viewlist, "r") as f:
                    tmp = [x.strip().split() for x in f.readlines()]
                viewlist = {
                    x[0] + "/" + x[1]: list(map(int, x[2:]))
                    for x in tmp
                }
            model = models.LFAutoDecoder(num_instances=num_instances, latent_dim=256, parameterization='plucker',
                                         network=opt.network,
                                         conditioning=opt.conditioning).cuda()
            model.eval()
            print("Loading retrain model")
            model.load_state_dict(state_dict)
    
            print("Loading dataset")
            if opt.dataset == 'NMR':
                dataset = multiclass_dataio.get_instance_datasets(opt.data_root, sidelen=opt.img_sidelength,
                                                                  dataset_type='test',
                                                                  max_num_instances=opt.max_num_instances)
    
            log_dir = Path(opt.logging_root) / opt.test_experiment_name
    
            if opt.dataset == 'NMR':
                class_psnrs = defaultdict(list)
                class_counter = defaultdict(int)
    
            with torch.no_grad():
                for i in range(len(dataset)):
                    
    
                    dummy_query = dataset[i][0]
                    instance_name = dummy_query['instance_name']
                    print(f"instance nmae {instance_name}")
                    if opt.dataset == 'NMR':
                        obj_class = int(dummy_query['class'].cpu().numpy())
                        obj_class = multiclass_dataio.class2string_dict[obj_class]
    
                        if class_counter[obj_class] < opt.save_out_first_n:
                            instance_dir = log_dir / f'{obj_class}' / f'{instance_name}'
                            instance_dir.mkdir(exist_ok=True, parents=True)
    
                    for j, query in enumerate(dataset[i]):
                        model_input = util.assemble_model_input(query, query)
                        model_output = model(model_input)
    
                        out_dict = {}
                        out_dict['rgb'] = model_output['rgb']
                        out_dict['gt_rgb'] = model_input['query']['rgb']
    
                        is_context = False
                        if opt.viewlist is not None:
                            key = '/'.join((obj_class, instance_name))
                            if key in viewlist:
                                if j in viewlist[key]:
                                    is_context = True
                            else:
                                print(f'{key} not in viewlist')
                                continue
    
                        # if opt.dataset != 'NMR' or not is_context:
                        psnr, ssim = get_psnr(out_dict['rgb'], out_dict['gt_rgb'])
                        if opt.dataset == 'NMR':
                            if not is_context:
                                class_psnrs[obj_class].append((psnr, ssim))
    
                        if opt.dataset == 'NMR' and class_counter[obj_class] < opt.save_out_first_n:
                            for k, v in out_dict.items():
                                img = convert_image(v, k)
                                if k == 'gt_rgb':
                                    cv2.imwrite(str(instance_dir / f"{j:06d}_{k}_{counter}_alsoinlier.png"),
                                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                                elif k == 'rgb':
                                    cv2.imwrite(str(instance_dir / f"{j:06d}_{counter}_alsoinlier.png"),
                                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        elif i < opt.save_out_first_n:
                            img = convert_image(out_dict['gt_rgb'], 'rgb')
                            cv2.imwrite(str(instance_dir / f"{j:06d}_gt.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                            img = convert_image(out_dict['rgb'], 'rgb')
                            cv2.imwrite(str(instance_dir / f"{j:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
                    if opt.dataset == 'NMR':
                        mean_dict = {}

                        for k, v in class_psnrs.items():
                            mean = np.mean(np.array(v), axis=0)
                            mean_dict[k] = f"{mean[0]:.3f} {mean[1]:.3f}"
                        print(f"mean_dict:{mean_dict}")
    
                        class_counter[obj_class] += 1
    
            with open(os.path.join(log_dir, f"results_alsoinlier_{counter}.txt"), "w") as out_file:
                if opt.dataset == 'NMR':
                    out_file.write(' & '.join(class_psnrs.keys()) + '\n')
    
                    psnrs, ssims = [], []
                    for value in class_psnrs.values():
                        mean = np.mean(np.array(value), axis=0)
                        psnrs.append(mean[0])
                        ssims.append(mean[1])
    
                    out_file.write(' & '.join(map(lambda x: f"{x:.3f}", psnrs)) + '\n')
                    out_file.write(' & '.join(map(lambda x: f"{x:.3f}", ssims)) + '\n')
            for (x,y) in zip(psnrs, ssims):
                correct_temp += 0.5 *(x+y)

        else:

            #print(f"class_psnrs:{class_psnrs}")
            for (x,y) in zip(psnrs, ssims):
                correct_temp += 0.5 *(x+y)
            print(f"return correct:{correct_temp}")
            return model,correct_temp,sparsity_list

    return model, correct_temp,sparsity_list


if __name__ == "__main__":

    ini_correct = 0
    best_model = None
    best_dict = None
    sparsity_best = None
    opt = p.parse_args()
    counter = 0
    rec_checkpoint_path = get_rec_model_dir()
    manager = Manager()
    shared_dict = manager.dict()
    
    print(f"This model has sparsity: {opt.sparsity}")
    for i in range(1):
        np.random.seed(i)
        random.seed(i)

        model, correct, sparsity_list = ransac(counter)
        if opt.sparsity is None:
            break        
        print(f"ini_correct and correct:{ini_correct} , {correct}")
        counter += 1
        if (correct >= ini_correct ):
            print(f"come into compare {counter}****")
            sparsity_best = sparsity_list
            best_model = model
            ini_correct = correct
            best_dict = torch.load(rec_checkpoint_path)
            print(f" correct change to: best_correct {correct}")

    rand_idcs = None
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt, shared_dict, rand_idcs, sparsity_best))
    else:
        multigpu_train(0, opt, shared_dict, rand_idcs, sparsity_best)



    if opt.viewlist is not None:
        with open(opt.viewlist, "r") as f:
            tmp = [x.strip().split() for x in f.readlines()]
        viewlist = {
            x[0] + "/" + x[1]: list(map(int, x[2:]))
            for x in tmp
        }

    rec_checkpoint_path = get_rec_model_dir()

    state_dict = torch.load(rec_checkpoint_path)
    print("Loading best model")
    model.load_state_dict(state_dict)

    print("Loading dataset")
    if opt.dataset == 'NMR':
        dataset = multiclass_dataio.get_instance_datasets(opt.data_root, sidelen=opt.img_sidelength,
                                                          dataset_type='test',
                                                          max_num_instances=opt.max_num_instances)
    else:
        dataset = hdf5_dataio.get_instance_datasets_hdf5(opt.data_root, sidelen=opt.img_sidelength,
                                                         max_num_instances=opt.max_num_instances)
    log_dir = Path(opt.logging_root) / opt.test_experiment_name

    if opt.dataset == 'NMR':
        class_psnrs = defaultdict(list)

        class_counter = defaultdict(int)
    else:
        psnrs = []
    with torch.no_grad():
        print(f"********len of dataset: {len(dataset)}****")
        for i in range(len(dataset)):
            

            dummy_query = dataset[i][0]
            instance_name = dummy_query['instance_name']
            print(f"instance nmae {instance_name}")
            if opt.dataset == 'NMR':
                obj_class = int(dummy_query['class'].cpu().numpy())
                obj_class = multiclass_dataio.class2string_dict[obj_class]

                if class_counter[obj_class] < opt.save_out_first_n:
                    instance_dir = log_dir / f'{obj_class}' / f'{instance_name}'
                    instance_dir.mkdir(exist_ok=True, parents=True)
            elif i < opt.save_out_first_n:
                instance_dir = log_dir / f'{instance_name}'
                instance_dir.mkdir(exist_ok=True, parents=True)

            for j, query in enumerate(dataset[i]):
                print(f"dataset{j} : {dataset[i]}")
                model_input = util.assemble_model_input(query, query)
                model_output = model(model_input)

                out_dict = {}
                out_dict['rgb'] = model_output['rgb']
                out_dict['gt_rgb'] = model_input['query']['rgb']

                is_context = False
                """
                    here the 02691156 4a559 will remove the value equal to j, which means remove the src_dvr.txt value
                """
                
                if opt.viewlist is not None:
                    key = '/'.join((obj_class, instance_name))
                    if key in viewlist:
                        if j in viewlist[key]:
                            is_context = True
                    else:
                        print(f'{key} not in viewlist')
                        continue
                
                # if opt.dataset != 'NMR' or not is_context:
                psnr, ssim = get_psnr(out_dict['rgb'], out_dict['gt_rgb'])
                if opt.dataset == 'NMR':
                    if not is_context:
                        #print(f"}add{psnr,ssim}")
                        class_psnrs[obj_class].append((psnr, ssim))
                else:
                    psnrs.append((psnr, ssim))

                if opt.dataset == 'NMR' and class_counter[obj_class] < opt.save_out_first_n:
                    for k, v in out_dict.items():
                        img = convert_image(v, k)
                        if k == 'gt_rgb':
                            cv2.imwrite(str(instance_dir / f"{j:06d}_{k}_final.png"),
                                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        elif k == 'rgb':
                            cv2.imwrite(str(instance_dir / f"{j:06d}_final.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                elif i < opt.save_out_first_n:
                    img = convert_image(out_dict['gt_rgb'], 'rgb')
                    cv2.imwrite(str(instance_dir / f"{j:06d}_gt_2.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    img = convert_image(out_dict['rgb'], 'rgb')
                    cv2.imwrite(str(instance_dir / f"{j:06d}_2.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            if opt.dataset == 'NMR':
                mean_dict = {}
                print(f"class_psnrs:{class_psnrs}")
                for k, v in class_psnrs.items():
                    mean = np.mean(np.array(v), axis=0)
                    mean_dict[k] = f"{mean[0]:.3f} {mean[1]:.3f}"
                print(f"mean_dict:{mean_dict}")

                class_counter[obj_class] += 1
            else:
                print(np.mean(np.array(psnrs), axis=0))

    with open(os.path.join(log_dir, "final_results.txt"), "w") as out_file:
        if opt.dataset == 'NMR':
            out_file.write(' & '.join(class_psnrs.keys()) + '\n')

            psnrs, ssims = [], []
            for value in class_psnrs.values():
                mean = np.mean(np.array(value), axis=0)
                psnrs.append(mean[0])
                ssims.append(mean[1])

            out_file.write(' & '.join(map(lambda x: f"{x:.3f}", psnrs)) + '\n')
            out_file.write(' & '.join(map(lambda x: f"{x:.3f}", ssims)) + '\n')
        else:
            mean = np.mean(psnrs, axis=0)
            out_file.write(f"{mean[0]} PSRN {mean[1]} SSIM")
