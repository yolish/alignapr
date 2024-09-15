"""
Entry point training and testing multi-scene transformer
"""
import argparse
import torch
import torch.nn as nn 
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.losses import CameraPoseLoss
from models.align2apr import Mapper, align_to_apr
from os.path import join


def main(args):
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Set the encoder 
    #TODO: make configurable to use b2q (cvpr2024) and others 
    encoder = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=config["encoder"]["output_dim"])
    
    # Initialize the alignment model
    mapper_config = config["mapper"]
    mapper["input_dim"] = config["encoder"]["output_dim"]
    mapper = Mapper(config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        mapper.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        mapper.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config["pose_loss"]).to(device)
        align_loss = None
        train_config = config["training"]
        if train_config["align"]:
            align_loss = nn.CrossEntropyLoss()
            alpha = train_config["alpha"]
        
        # Set the optimizer and scheduler
        params = list(mapper.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=train_config['lr'],
                                  eps=train_config['eps'],
                                  weight_decay=train_config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=train_config['lr_scheduler_step_size'],
                                                    gamma=train_config['lr_scheduler_gamma'])

        # Set the dataset and data loader
        transform = utils.train_transforms.get('baseline') #TODO check this 
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        loader_params = {'batch_size': train_config['batch_size'],
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = train_config["n_freq_print"]
        n_freq_checkpoint = train_config["n_freq_checkpoint"]
        n_epochs = train_config["n_epochs"]

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0
            for batch_idx, data in enumerate(dataloader):
                # TODO connect to tensor board or wandb 
                res = train_step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    batch_loss = res["total_loss"]
                    epoch_loss += res["total_loss"]
                    # TODO add alignment and pose loss 
                    posit_err, orient_err = utils.pose_err(res["est_poses"], res["poses"])
                    logging.info("[Batch-{}/Epoch-{}] batch loss: {:.3f}, epoch loss: {:.3f},"
                                    "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, batch_loss, epoch_loss,
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(mapper.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

    else: # Test
        # Set to eval mode
        mapper.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):                
                
                posit_err, orient_err, runtime = test_step(data, mapper, encoder, pose_loss, device)
                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = runtime

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))

        # TODO revisit nan issue
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))

def train_step(data, mapper, encoder, optimizer, pose_loss, device, align_loss=None, alpha=1.0):
    poses = data['pose']
    imgs = data['img'].to(device)
    batch_size = poses.shape[0]

    # Zero the gradients
    optimizer.zero_grad()
    res = align_to_apr(imgs, encoder, mapper)
    aligned_features = res.get('features')
    est_poses = res.get('pose')
    pose_criterion = pose_loss(est_poses, poses)
    ret_val = {"batch_size":batch_size,
               "est_poses": est_poses.detach(),
               "poses":poses.detach(),
                "pose_loss":pose_criterion.item()}
    
    if align_loss is not None:
        features_sim_mat = get_features_sim_mat(res["features"])
        poses_sim_mat = get_pose_sim_mat(poses, pose_loss)
        alignment_criterion = align_loss(features_sim_mat, poses_sim_mat)
        ret_val["align_loss"] = alignment_criterion.item()
        criterion = pose_criterion + alpha*alignment_criterion
    else:
        # Pose loss
        criterion = pose_criterion 
    ret_val["total_loss"] = criterion.item()
    
    # Back prop
    criterion.backward()
    optimizer.step()
    return ret_val

def get_features_sim_mat(features):
    # TODO extend to other similarity metrics
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sim_mat = cos(features, features)
    sim_mat.fill_diagonal_(-2)
    return sim_mat


def get_pose_sim_mat(poses, pose_loss):
    n = poses.shape[0]
    sim_mat = torch.zeros((n,n)).astype(poses.dtype).to(poses.device)
    with torch.no_grad():
        for i in range(n):
            sim_mat[i, :] = -pose_loss(poses[i].repeat(n), poses)
    sim_mat.fill_diagonal_(torch.min(sim_mat))
    return sim_mat

def test_step(data, mapper, encoder, pose_loss, device):
    poses = data.get('pose').to(dtype=torch.float32)
    imgs = data.get('img').to(device)

    # Forward pass to predict the pose
    
    poses = data['pose']
    imgs = data['img'].to(device)
    batch_size = poses.shape[0]

    tic = time.time()
    res = align_to_apr(imgs, encoder, mapper)
    toc = time.time()

    # Evaluate error
    posit_err, orient_err = utils.pose_err(res["est_poses"], poses)
    return posit_err, orient_err, (toc-tic)*1000


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or test")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    main(args)