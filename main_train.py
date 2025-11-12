import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from networks.MSFPNet.network_backbone import MSFP
from load_datasets_transforms import data_loader, data_transforms
from loss import DC_and_CE_loss

def performValidation(model, valLoader, postLabel, postPred, diceMetric, writer, globalStep, device):
    model.eval()
    diceVals = []
    with torch.no_grad():
        epochIteratorVal = tqdm(valLoader, desc="Validating", dynamic_ncols=True)
        for batch in epochIteratorVal:
            valInputs = batch["image"].to(device)
            valLabels = batch["label"].to(device)

            valOutputs = sliding_window_inference(valInputs, roi_size=(128, 128, 96), sw_batch_size=2, predictor=model)

            valLabelsList = decollate_batch(valLabels)
            valLabelsConvert = [postLabel(label) for label in valLabelsList]

            valOutputsList = decollate_batch(valOutputs)
            valOutputConvert = [postPred(output) for output in valOutputsList]

            diceMetric(y_pred=valOutputConvert, y=valLabelsConvert)
            dice = diceMetric.aggregate().item()
            diceVals.append(dice)
            diceMetric.reset()

    meanDiceVal = np.mean(diceVals)
    writer.add_scalar('Validation Segmentation Loss', meanDiceVal, globalStep)
    return meanDiceVal


def trainModel(
    model, trainLoader, valLoader, optimizer, scheduler, lossFunction,
    postLabel, postPred, diceMetric, writer, rootDir, outputFilePath,
    maxIterations, evalStep, device
):
    globalStep = 0
    diceValBest = 0.0
    globalStepBest = 0
    stepCount = 0
    averageLoss = 0.0
    model.train()

    while globalStep < maxIterations:
        epochIterator = tqdm(trainLoader, desc="Training", dynamic_ncols=True)
        for batch in epochIterator:
            if globalStep >= maxIterations:
                break

            x = batch["image"].to(device)
            y = batch["label"].to(device)

            outputs = model(x)
            loss = lossFunction(outputs, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            currentLr = scheduler.get_last_lr()[0]
            averageLoss += loss.item()
            stepCount += 1

            epochIterator.set_description(f"Training ({globalStep}/{maxIterations}) (loss={loss:.5f}, lr={currentLr:.2e})")

            # Log average loss every 48 steps
            if stepCount == 48:
                avgLoss = averageLoss / 48
                with open(outputFilePath, "a") as f:
                    f.write(f"{globalStep} steps: Average loss for last 48 steps: {avgLoss}\n")
                writer.add_scalar('Average Training Loss', avgLoss, globalStep)
                averageLoss = 0.0
                stepCount = 0

            # Validation and checkpointing
            if (globalStep % evalStep == 0 and globalStep != 0) or globalStep == maxIterations - 1:
                diceVal = performValidation(model, valLoader, postLabel, postPred, diceMetric, writer, globalStep, device)

                if diceVal > diceValBest:
                    diceValBest = diceVal
                    globalStepBest = globalStep
                    torch.save(model.state_dict(), os.path.join(rootDir, "best_metric_model.pth"))
                    print(f"Model saved! Best Dice: {diceValBest:.5f}")
                    with open(outputFilePath, "a") as f:
                        f.write(f"{globalStep} Validate Saving!!!: {diceVal}\n")
                else:
                    print(f"Model not saved. Best Dice: {diceValBest:.5f}, Current: {diceVal:.5f}")
                    with open(outputFilePath, "a") as f:
                        f.write(f"{globalStep} Validate: {diceVal}\n")

            writer.add_scalar('Training Segmentation Loss', loss.item(), globalStep)
            globalStep += 1

    print(f"Training finished. Best validation Dice: {diceValBest:.5f} at step {globalStepBest}")


def main():
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')

    # Model & training args
    parser.add_argument('--network', type=str, default='MSFP')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrained_weights', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--crop_sample', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--max_iter', type=int, default=40000)
    parser.add_argument('--eval_step', type=int, default=300)

    # Efficiency args
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cache_rate', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f'Using GPU: {args.gpu}')

    # Set seeds
    set_determinism(seed=0)

    # Load data
    trainSamples, validSamples, outClasses = data_loader(args)

    trainFiles = [{"image": im, "label": lb} for im, lb in zip(trainSamples['images'], trainSamples['labels'])]
    valFiles = [{"image": im, "label": lb} for im, lb in zip(validSamples['images'], validSamples['labels'])]

    trainTransforms, valTransforms = data_transforms(args)

    trainDs = CacheDataset(data=trainFiles, transform=trainTransforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    trainLoader = DataLoader(trainDs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valDs = CacheDataset(data=valFiles, transform=valTransforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    valLoader = DataLoader(valDs, batch_size=1, num_workers=args.num_workers)

    # Initialize model
    device = torch.device("cuda:0")
    model = MSFP(in_chans=1, out_chans=outClasses, feat_size=[64, 128, 256, 512]).to(device)

    print(f'Chosen Network Architecture: {args.network}')

    if args.pretrain:
        if os.path.exists(args.pretrained_weights):
            model.load_state_dict(torch.load(args.pretrained_weights))
            print(f'Loaded pretrained weights from: {args.pretrained_weights}')
        else:
            print(f'Pretrained weights not found at {args.pretrained_weights}. Training from scratch.')

    # Loss & optimizer
    lossFunction = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    print('Loss for training: DiceCELoss')

    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optim}")

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iter)
    print(f'Optimizer: {args.optim}, Learning rate: {args.lr}')

    # Setup directories & logging
    rootDir = os.path.join(args.output)
    os.makedirs(rootDir, exist_ok=True)

    tDir = os.path.join(rootDir, 'tensorboard')
    os.makedirs(tDir, exist_ok=True)
    writer = SummaryWriter(log_dir=tDir)

    outputFilePath = os.path.join(rootDir, "average_loss.txt")

    # Metrics
    postLabel = AsDiscrete(to_onehot=outClasses)
    postPred = AsDiscrete(argmax=True, to_onehot=outClasses)
    diceMetric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Start training
    trainModel(
        model=model,
        trainLoader=trainLoader,
        valLoader=valLoader,
        optimizer=optimizer,
        scheduler=scheduler,
        lossFunction=lossFunction,
        postLabel=postLabel,
        postPred=postPred,
        diceMetric=diceMetric,
        writer=writer,
        rootDir=rootDir,
        outputFilePath=outputFilePath,
        maxIterations=args.max_iter,
        evalStep=args.eval_step,
        device=device
    )


if __name__ == "__main__":
    main()